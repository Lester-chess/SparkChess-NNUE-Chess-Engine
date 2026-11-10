/**
 * @file nnue.hpp
 * @brief Valkyrie C++20 Chess Engine - Stage 3: HalfKAv2 512x2 NNUE Evaluation Network,
 *        23,107,844-Byte Binary Loader, Dual-Perspective Incremental Accumulator,
 *        Pairwise Clipped-ReLU (>> 9), 4-Bucket LayerStacks (fc_0/fc_1/fc_2 + l1_skip).
 * @standard C++20 (-std=c++20 / /std:c++20)
 * @license MIT (100% Original Architecture designed for HalfKAv2 512x2 NNUE)
 */

#pragma once

#include "position.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <immintrin.h>

namespace valkyrie::nnue {

// ============================================================================
// 1. 网络拓扑常量、Magic 与 Hash 校验码 (严格对标 23,107,844 字节正式训练网络)
// ============================================================================

inline constexpr uint32_t kVersion     = 0x6A448AFAU;
inline constexpr uint32_t kFeatureHash = 0x5F134CB8U;
inline constexpr uint32_t kFtHash      = 0x5F1348B8U; // kFeatureHash ^ (512 * 2)
inline constexpr uint32_t kFcHash      = 0x633376CAU;
inline constexpr uint32_t kNetHash     = 0x3C203E72U; // kFtHash ^ kFcHash

inline constexpr size_t kFtInputDims   = 45056;       // 64 王格桶 x 11 平面 x 64 格
inline constexpr size_t kFtHalfDims    = 512;         // 单视角特征变换器输出维度
inline constexpr size_t kFtPairwiseOut = 256;         // 单视角成对相乘后维度 (512 / 2)
inline constexpr size_t kFc0InDims     = 512;         // Us(256) + Them(256) 拼接
inline constexpr size_t kFc0OutDims    = 16;          // fc_0 输出维度 (前14维或全16维参与激活，14/15提供 l1_skip)
inline constexpr size_t kFc1InDims     = 32;          // [ (fc0_out^2) >> 7, fc0_out ] 双激活拼接 (16 * 2)
inline constexpr size_t kFc1OutDims    = 32;          // fc_1 输出维度
inline constexpr size_t kFc2InDims     = 96;          // fc_0 双激活(32维) + fc_1 双激活(64维) 残差拼接 = 96 维
inline constexpr size_t kNumLsBuckets  = 4;           // 4 个子力分桶 (LayerStacks)

inline constexpr size_t kExpectedFileSize = 23107844ULL;
inline constexpr double kNnue2Score       = 600.0;

// 量化缩放系数（严格对标 PyTorch 训练器 layer_stacks.py 的量化尺度）
inline constexpr int32_t kFc0WeightShift = 7;     // fc_0 输出 >> 7 (权重缩放 128)
inline constexpr int32_t kFc1WeightShift = 6;     // fc_1 输出 >> 6 (权重缩放 64)
inline constexpr int32_t kSkipScaleShift = 7;     // l1_skip(尺度 128) << 7 对齐至 fc2_sum(尺度 16384)
inline constexpr int64_t kOutputScale    = 16384; // 128 * 128 = 16,384
inline constexpr int64_t kNnue2ScoreInt  = 600;   // cp = (combined * 600) / 16384

// 根据盘面总棋子数动态选择 LayerStack 分桶 [0..3]
[[nodiscard]] constexpr int select_layer_stack_bucket(int piece_count) noexcept {
    return std::clamp((piece_count - 1) / 8, 0, 3);
}

// ============================================================================
// 2. 双视角增量累加器 (Accumulator) 与 DirtyPiece 差异记录
// ============================================================================

struct alignas(64) Accumulator {
    alignas(64) std::array<int16_t, kFtHalfDims> white{};
    alignas(64) std::array<int16_t, kFtHalfDims> black{};
    Square white_king_sq = Square::None;
    Square black_king_sq = Square::None;
    bool   computed_white = false;
    bool   computed_black = false;

    [[nodiscard]] constexpr std::array<int16_t, kFtHalfDims>& operator[](Color c) noexcept {
        return (c == Color::White) ? white : black;
    }

    [[nodiscard]] constexpr const std::array<int16_t, kFtHalfDims>& operator[](Color c) const noexcept {
        return (c == Color::White) ? white : black;
    }
};

// 记录单步 make_move / undo_move 引起的特征增删（最多移除 2 子 + 新增 2 子，如王车易位或吃过路兵）
struct DirtyFeatures {
    static constexpr size_t kMaxChanged = 2;

    struct PieceSq {
        Piece  piece = Piece::None;
        Square sq    = Square::None;
    };

    std::array<PieceSq, kMaxChanged> removed{};
    std::array<PieceSq, kMaxChanged> added{};
    size_t num_removed    = 0;
    size_t num_added      = 0;
    bool   white_king_moved = false;
    bool   black_king_moved = false;

    constexpr void add_removed(Piece p, Square s) noexcept {
        assert(num_removed < kMaxChanged);
        removed[num_removed++] = PieceSq{p, s};
    }

    constexpr void add_added(Piece p, Square s) noexcept {
        assert(num_added < kMaxChanged);
        added[num_added++] = PieceSq{p, s};
    }
};

// 从 (Position, Move, StateInfo) 提取精确的 DirtyFeatures（零分支遗漏涵盖升变、易位、吃过路兵）
[[nodiscard]] constexpr DirtyFeatures extract_dirty_features(
    Color mover_color,
    Piece moved_piece_before,
    Move m,
    const StateInfo& st) noexcept
{
    DirtyFeatures df{};
    const Square   from = m.from_sq();
    const Square   to   = m.to_sq();
    const MoveFlag fl   = m.flags();

    if (piece_type(moved_piece_before) == PieceType::King) {
        if (mover_color == Color::White) df.white_king_moved = true;
        else                             df.black_king_moved = true;
    }

    // 1. 起点棋子移除
    df.add_removed(moved_piece_before, from);

    // 2. 终点棋子新增（考虑升变）
    const Piece placed = m.is_promotion()
        ? make_piece(mover_color, m.promotion_type())
        : moved_piece_before;
    df.add_added(placed, to);

    // 3. 被吃子移除（常规吃子或吃过路兵）
    if (fl == MoveFlag::EnPassantCapture) {
        const Square cap_sq = (mover_color == Color::White)
            ? static_cast<Square>(sq_index(to) - 8U)
            : static_cast<Square>(sq_index(to) + 8U);
        df.add_removed(st.captured_piece, cap_sq);
    } else if (st.captured_piece != Piece::None) {
        df.add_removed(st.captured_piece, to);
    }

    // 4. 王车易位时车的移动
    if (fl == MoveFlag::KingCastle) {
        const Square r_from = (mover_color == Color::White) ? Square::h1 : Square::h8;
        const Square r_to   = (mover_color == Color::White) ? Square::f1 : Square::f8;
        const Piece  rook   = make_piece(mover_color, PieceType::Rook);
        df.add_removed(rook, r_from);
        df.add_added(rook, r_to);
    } else if (fl == MoveFlag::QueenCastle) {
        const Square r_from = (mover_color == Color::White) ? Square::a1 : Square::a8;
        const Square r_to   = (mover_color == Color::White) ? Square::d1 : Square::d8;
        const Piece  rook   = make_piece(mover_color, PieceType::Rook);
        df.add_removed(rook, r_from);
        df.add_added(rook, r_to);
    }

    return df;
}

// ============================================================================
// 3. LayerStack 子力分桶结构体与 Network 完整参数存储
// ============================================================================

struct alignas(64) LayerStackBucket {
    // fc_0: 512 -> 16
    alignas(64) std::array<int32_t, kFc0OutDims>               fc0_biases{};  // 64 B
    alignas(64) std::array<int8_t,  kFc0OutDims * kFc0InDims>  fc0_weights{}; // 8,192 B

    // fc_1: 32 -> 32
    alignas(64) std::array<int32_t, kFc1OutDims>               fc1_biases{};  // 128 B
    alignas(64) std::array<int8_t,  kFc1OutDims * kFc1InDims>  fc1_weights{}; // 1,024 B

    // fc_2: 96 -> 1 (前 32 维 fc0 双激活 + 后 64 维 fc1 双激活拼接)
    int32_t                                                    fc2_bias = 0;  // 4 B
    alignas(64) std::array<int8_t,  kFc2InDims>                fc2_weights{}; // 96 B
};

struct NetworkParameters {
    // Feature Transformer: 偏置 int16_t[512] (1,024 B) + 权重 int8_t[45056 * 512] (23,068,672 B)
    alignas(64) std::array<int16_t, kFtHalfDims>               ft_biases{};
    alignas(64) std::array<int8_t,  kFtInputDims * kFtHalfDims> ft_weights{};

    // 4 个子力分桶 (LayerStacks)
    std::array<LayerStackBucket, kNumLsBuckets>                buckets{};
};

// ============================================================================
// ============================================================================
// 3.5 现代全套残局缩放因子体系 (Endgame Scale Factor System - 严格对标 Stockfish)
//     覆盖：KNNK双马、材质不足、错色象边兵死要塞(KBPK)、纯异色格象(OCB)、
//           带重子异色格象(KRB v KRB / KQB v KQB)、车单兵对车(KRPKR菲利多尔/边兵)
// ============================================================================
[[nodiscard]] inline int calculate_endgame_scale_factor(const Position& pos) noexcept {
    const Bitboard w_pawns   = pos.pieces(Color::White, PieceType::Pawn);
    const Bitboard b_pawns   = pos.pieces(Color::Black, PieceType::Pawn);
    const Bitboard w_knights = pos.pieces(Color::White, PieceType::Knight);
    const Bitboard b_knights = pos.pieces(Color::Black, PieceType::Knight);
    const Bitboard w_bishops = pos.pieces(Color::White, PieceType::Bishop);
    const Bitboard b_bishops = pos.pieces(Color::Black, PieceType::Bishop);
    const Bitboard w_rooks   = pos.pieces(Color::White, PieceType::Rook);
    const Bitboard b_rooks   = pos.pieces(Color::Black, PieceType::Rook);
    const Bitboard w_queens  = pos.pieces(Color::White, PieceType::Queen);
    const Bitboard b_queens  = pos.pieces(Color::Black, PieceType::Queen);

    const int w_p_count = w_pawns.popcount();
    const int b_p_count = b_pawns.popcount();
    const int total_pawns = w_p_count + b_p_count;

    // ------------------------------------------------------------------------
    // (A) 无兵特殊残局（材质不足 / 双马对孤王）
    // ------------------------------------------------------------------------
    if (total_pawns == 0) {
        if (w_rooks.empty() && b_rooks.empty() && w_queens.empty() && b_queens.empty()) {
            const int w_minors = w_knights.popcount() + w_bishops.popcount();
            const int b_minors = b_knights.popcount() + b_bishops.popcount();

            // KBK / KNK / KBKB (同或异色单轻子对单轻子): 材质不足绝对和棋
            if (w_minors <= 1 && b_minors <= 1) return 0;

            // KNNK (双马对单王): 国际象棋规则下双马无法对孤王强制杀棋
            if ((w_knights.popcount() == 2 && w_bishops.empty() && b_minors == 0) ||
                (b_knights.popcount() == 2 && b_bishops.empty() && w_minors == 0))
            {
                return 0;
            }

            // KNNKB (双马对单轻子): 理论必和
            if ((w_knights.popcount() == 2 && w_bishops.empty() && b_minors <= 1) ||
                (b_knights.popcount() == 2 && b_bishops.empty() && w_minors <= 1))
            {
                return 4;
            }
        }
    }

    // ------------------------------------------------------------------------
    // (B) 错色象与边兵理论和棋 (Wrong-Colored Bishop & Rook Pawn, KBPK)
    // ------------------------------------------------------------------------
    {
        auto check_wrong_bishop = [&](Color superior, Color inferior) -> bool {
            if (pos.pieces(inferior, PieceType::Pawn).any() ||
                pos.pieces(inferior, PieceType::Knight).any() ||
                pos.pieces(inferior, PieceType::Bishop).any() ||
                pos.pieces(inferior, PieceType::Rook).any()   ||
                pos.pieces(inferior, PieceType::Queen).any())
            {
                return false;
            }
            if (pos.pieces(superior, PieceType::Knight).any() ||
                pos.pieces(superior, PieceType::Rook).any()   ||
                pos.pieces(superior, PieceType::Queen).any()  ||
                pos.pieces(superior, PieceType::Bishop).popcount() != 1)
            {
                return false;
            }

            const Bitboard sup_pawns = pos.pieces(superior, PieceType::Pawn);
            if (sup_pawns.empty()) return false;

            constexpr Bitboard a_file_bb(0x0101010101010101ULL);
            constexpr Bitboard h_file_bb(0x8080808080808080ULL);

            const bool all_a = (sup_pawns & ~a_file_bb).empty();
            const bool all_h = (sup_pawns & ~h_file_bb).empty();
            if (!all_a && !all_h) return false;

            const Square queening_sq = all_a
                ? (superior == Color::White ? Square::a8 : Square::a1)
                : (superior == Color::White ? Square::h8 : Square::h1);

            const Square bishop_sq = pos.pieces(superior, PieceType::Bishop).lsb();
            const bool b_dark = ((file_of(bishop_sq) + rank_of(bishop_sq)) & 1) == 0;
            const bool q_dark = ((file_of(queening_sq) + rank_of(queening_sq)) & 1) == 0;

            if (b_dark == q_dark) return false;

            const Square inf_ksq = pos.pieces(inferior, PieceType::King).lsb();
            const int king_dist = std::max(
                std::abs(static_cast<int>(file_of(inf_ksq)) - static_cast<int>(file_of(queening_sq))),
                std::abs(static_cast<int>(rank_of(inf_ksq)) - static_cast<int>(rank_of(queening_sq))));

            return (king_dist <= 2);
        };

        if (check_wrong_bishop(Color::White, Color::Black) ||
            check_wrong_bishop(Color::Black, Color::White))
        {
            return 0; // 错色象边兵死要塞：直接判 0 分理论和棋！
        }
    }

    // ------------------------------------------------------------------------
    // (C) 纯异色格象残局 (Opposite-Colored Bishops + Pawns)
    // ------------------------------------------------------------------------
    if (w_knights.empty() && b_knights.empty() &&
        w_rooks.empty()   && b_rooks.empty()   &&
        w_queens.empty()  && b_queens.empty()  &&
        w_bishops.popcount() == 1 && b_bishops.popcount() == 1)
    {
        const Square w_bsq = w_bishops.lsb();
        const Square b_bsq = b_bishops.lsb();
        const bool w_dark = ((file_of(w_bsq) + rank_of(w_bsq)) & 1) == 0;
        const bool b_dark = ((file_of(b_bsq) + rank_of(b_bsq)) & 1) == 0;

        if (w_dark != b_dark) {
            const int pawn_diff = std::abs(w_p_count - b_p_count);
            const Square w_ksq = pos.pieces(Color::White, PieceType::King).lsb();
            const Square b_ksq = pos.pieces(Color::Black, PieceType::King).lsb();

            int locked_pairs = 0;
            for (Square s : b_pawns) {
                const Square blocker = static_cast<Square>(sq_index(s) - 8U);
                if (sq_index(blocker) < 64U && w_pawns.test(blocker)) {
                    ++locked_pairs;
                }
            }

            bool adv_pawn_blockaded = false;
            if (b_pawns.test(Square::g2) && w_ksq == Square::g1) adv_pawn_blockaded = true;
            if (b_pawns.test(Square::h2) && w_ksq == Square::h1) adv_pawn_blockaded = true;
            if (b_pawns.test(Square::a2) && w_ksq == Square::a1) adv_pawn_blockaded = true;
            if (b_pawns.test(Square::b2) && w_ksq == Square::b1) adv_pawn_blockaded = true;
            if (w_pawns.test(Square::g7) && b_ksq == Square::g8) adv_pawn_blockaded = true;
            if (w_pawns.test(Square::h7) && b_ksq == Square::h8) adv_pawn_blockaded = true;
            if (w_pawns.test(Square::a7) && b_ksq == Square::a8) adv_pawn_blockaded = true;
            if (w_pawns.test(Square::b7) && b_ksq == Square::b8) adv_pawn_blockaded = true;

            if (pawn_diff == 0) {
                if (locked_pairs >= 1 || adv_pawn_blockaded) return 0; // 兵数均等且死锁：绝对死和棋 (归零)
                if (total_pawns <= 4) return 2;
                return 8;
            }
            if (pawn_diff == 1) {
                if (locked_pairs >= 2) return 0; // 至少两对兵死锁：不可突破的绝对死要塞，对标 Stockfish 精确判 0
                if (locked_pairs >= 1 && adv_pawn_blockaded) return 0;
                if (locked_pairs >= 1) return 8;
                return 16;
            }
            if (pawn_diff == 2) {
                if (locked_pairs >= 3) return 0;
                if (locked_pairs >= 2) return 8;
                return 24;
            }
            return 40;
        }
    }

    // ------------------------------------------------------------------------
    // (D) 带重子的异色格象残局 (KRB vs KRB / KQB vs KQB)
    // ------------------------------------------------------------------------
    if (w_knights.empty() && b_knights.empty() &&
        w_bishops.popcount() == 1 && b_bishops.popcount() == 1)
    {
        const Square w_bsq = w_bishops.lsb();
        const Square b_bsq = b_bishops.lsb();
        const bool w_dark = ((file_of(w_bsq) + rank_of(w_bsq)) & 1) == 0;
        const bool b_dark = ((file_of(b_bsq) + rank_of(b_bsq)) & 1) == 0;

        if (w_dark != b_dark) {
            if (w_queens.empty() && b_queens.empty() && w_rooks.popcount() == 1 && b_rooks.popcount() == 1) {
                const int pawn_diff = std::abs(w_p_count - b_p_count);
                if (pawn_diff <= 1) return 36;
                if (pawn_diff == 2) return 48;
            }
            if (w_rooks.empty() && b_rooks.empty() && w_queens.popcount() == 1 && b_queens.popcount() == 1) {
                const int pawn_diff = std::abs(w_p_count - b_p_count);
                if (pawn_diff <= 1) return 40;
            }
        }
    }

    // ------------------------------------------------------------------------
    // (E) 单车单兵对单车残局 (KRPKR - 菲利多尔防御与边兵死和)
    // ------------------------------------------------------------------------
    if (w_queens.empty() && b_queens.empty() &&
        w_knights.empty() && b_knights.empty() &&
        w_bishops.empty() && b_bishops.empty() &&
        w_rooks.popcount() == 1 && b_rooks.popcount() == 1)
    {
        auto check_krpkr = [&](Color superior, Color inferior) -> int {
            if (pos.pieces(inferior, PieceType::Pawn).any() ||
                pos.pieces(superior, PieceType::Pawn).popcount() != 1)
            {
                return 64;
            }

            const Square pawn_sq = pos.pieces(superior, PieceType::Pawn).lsb();
            const Square inf_ksq = pos.pieces(inferior, PieceType::King).lsb();
            const uint8_t p_file = file_of(pawn_sq);
            const uint8_t k_file = file_of(inf_ksq);

            if (p_file == 0U || p_file == 7U) {
                const Square q_sq = (p_file == 0U)
                    ? (superior == Color::White ? Square::a8 : Square::a1)
                    : (superior == Color::White ? Square::h8 : Square::h1);
                const int k_dist = std::max(
                    std::abs(static_cast<int>(file_of(inf_ksq)) - static_cast<int>(file_of(q_sq))),
                    std::abs(static_cast<int>(rank_of(inf_ksq)) - static_cast<int>(rank_of(q_sq))));
                if (k_dist <= 2) return 8;
            }

            if (p_file == k_file) {
                const uint8_t p_rank = rank_of(pawn_sq);
                const uint8_t k_rank = rank_of(inf_ksq);
                const bool king_in_front = (superior == Color::White) ? (k_rank > p_rank) : (k_rank < p_rank);
                if (king_in_front) {
                    return 16;
                }
            }

            return 64;
        };

        const int s_w = check_krpkr(Color::White, Color::Black);
        if (s_w < 64) return s_w;
        const int s_b = check_krpkr(Color::Black, Color::White);
        if (s_b < 64) return s_b;
    }

    return 64;
}

// 4. Evaluator 评估网络核心类 (支持二进制加载、增量累加器维护、前向推理)
// ============================================================================

class Evaluator {
public:
    Evaluator() : params_(std::make_unique<NetworkParameters>()) {
        init_deterministic_test_weights();
    }

    // 初始化确定性非零测试权重（当未加载外部 .nnue 文件时可用于自检与增量一致性验证）
    void init_deterministic_test_weights() noexcept {
        uint64_t rng = 0x3C203E726A448AFAULL;
        auto next_val = [&]() noexcept -> uint64_t {
            rng ^= rng >> 12U;
            rng ^= rng << 25U;
            rng ^= rng >> 27U;
            return rng * 2685821657736338717ULL;
        };

        for (size_t i = 0; i < kFtHalfDims; ++i) {
            params_->ft_biases[i] = static_cast<int16_t>((next_val() % 65ULL) - 32ULL);
        }
        // 仅抽样初始化部分 FT 权重以保证启动瞬时完成，同时赋予各子力合理梯度
        std::memset(params_->ft_weights.data(), 0, params_->ft_weights.size());
        constexpr int16_t kPieceBase[6] = {12, 34, 36, 52, 96, 0};
        for (size_t ksq = 0; ksq < 64; ++ksq) {
            for (size_t plane = 0; plane < 11; ++plane) {
                const int sign = (plane < 10 && (plane % 2 == 1)) ? -1 : 1;
                const int pt   = (plane < 10) ? static_cast<int>(plane / 2) : 5;
                const int base = sign * kPieceBase[pt];
                for (size_t sq = 0; sq < 64; ++sq) {
                    const size_t feat_idx = sq + plane * 64 + ksq * 704;
                    int8_t* w_row = &params_->ft_weights[feat_idx * kFtHalfDims];
                    const int center_bonus = 7 - static_cast<int>(
                        std::abs(static_cast<int>(sq & 7) - 3) + std::abs(static_cast<int>(sq >> 3) - 3));
                    for (size_t d = 0; d < kFtHalfDims; ++d) {
                        const int jitter = static_cast<int>(((feat_idx * 131 + d * 17) & 7)) - 3;
                        w_row[d] = static_cast<int8_t>(std::clamp(base + sign * center_bonus + jitter, -127, 127));
                    }
                }
            }
        }

        for (size_t b = 0; b < kNumLsBuckets; ++b) {
            auto& ls = params_->buckets[b];
            for (size_t i = 0; i < kFc0OutDims; ++i) {
                ls.fc0_biases[i] = static_cast<int32_t>((next_val() % 256ULL) - 128ULL);
            }
            for (size_t i = 0; i < ls.fc0_weights.size(); ++i) {
                ls.fc0_weights[i] = static_cast<int8_t>((next_val() % 33ULL) - 16ULL);
            }
            for (size_t i = 0; i < kFc1OutDims; ++i) {
                ls.fc1_biases[i] = static_cast<int32_t>((next_val() % 256ULL) - 128ULL);
            }
            for (size_t i = 0; i < ls.fc1_weights.size(); ++i) {
                ls.fc1_weights[i] = static_cast<int8_t>((next_val() % 33ULL) - 16ULL);
            }
            ls.fc2_bias = static_cast<int32_t>((next_val() % 128ULL) - 64ULL);
            for (size_t i = 0; i < kFc2InDims; ++i) {
                ls.fc2_weights[i] = static_cast<int8_t>((next_val() % 33ULL) - 16ULL);
            }
        }
    }

    // ------------------------------------------------------------------------
    // 二进制网络加载器：严格校验 23,107,844 字节结构与 VERSION / NET_HASH / FT_HASH / FC_HASH
    // ------------------------------------------------------------------------
    [[nodiscard]] bool load_network(const std::string& path) {
        network_loaded_ = false;
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) return false;

        const auto file_size = static_cast<size_t>(ifs.tellg());
        if (file_size != kExpectedFileSize) {
            // 若文件大小不等于 23,107,844 字节则拒绝加载
            return false;
        }
        ifs.seekg(0, std::ios::beg);

        auto read_u32 = [&](uint32_t& out) -> bool {
            return static_cast<bool>(ifs.read(reinterpret_cast<char*>(&out), sizeof(uint32_t)));
        };

        // 1. 校验 Header (Version = 0x6A448AFA, NetHash = 0x3C203E72, DescLen + Architecture String)
        uint32_t version = 0, net_hash = 0, desc_len = 0;
        if (!read_u32(version) || version != kVersion) return false;
        if (!read_u32(net_hash) || net_hash != kNetHash) return false;
        if (!read_u32(desc_len) || desc_len > 4096U) return false;

        std::string arch_desc(desc_len, '\0');
        if (desc_len > 0 && !ifs.read(arch_desc.data(), desc_len)) return false;

        // 2. 校验 Feature Transformer Header (FT_HASH = 0x5F1348B8)
        uint32_t ft_hash = 0;
        if (!read_u32(ft_hash) || ft_hash != kFtHash) return false;

        // 读取 FT Biases: int16_t[512] (1,024 bytes)
        if (!ifs.read(reinterpret_cast<char*>(params_->ft_biases.data()),
                      static_cast<std::streamsize>(params_->ft_biases.size() * sizeof(int16_t)))) {
            return false;
        }

        // 读取 FT Weights: int8_t[45056 * 512] (23,068,672 bytes)
        if (!ifs.read(reinterpret_cast<char*>(params_->ft_weights.data()),
                      static_cast<std::streamsize>(params_->ft_weights.size() * sizeof(int8_t)))) {
            return false;
        }

        // 3. 循环读取 4 个 LayerStack 子力分桶 (每个分桶带 FC_HASH = 0x633376CA)
        for (size_t b = 0; b < kNumLsBuckets; ++b) {
            uint32_t fc_hash = 0;
            if (!read_u32(fc_hash) || fc_hash != kFcHash) return false;

            auto& ls = params_->buckets[b];

            // fc_0: biases (int32_t[16]) + weights (int8_t[16 * 512])
            if (!ifs.read(reinterpret_cast<char*>(ls.fc0_biases.data()),
                          static_cast<std::streamsize>(ls.fc0_biases.size() * sizeof(int32_t)))) {
                return false;
            }
            if (!ifs.read(reinterpret_cast<char*>(ls.fc0_weights.data()),
                          static_cast<std::streamsize>(ls.fc0_weights.size() * sizeof(int8_t)))) {
                return false;
            }

            // fc_1: biases (int32_t[32]) + weights (int8_t[32 * 32])
            if (!ifs.read(reinterpret_cast<char*>(ls.fc1_biases.data()),
                          static_cast<std::streamsize>(ls.fc1_biases.size() * sizeof(int32_t)))) {
                return false;
            }
            if (!ifs.read(reinterpret_cast<char*>(ls.fc1_weights.data()),
                          static_cast<std::streamsize>(ls.fc1_weights.size() * sizeof(int8_t)))) {
                return false;
            }

            // fc_2: bias (int32_t[1]) + weights (int8_t[96])
            if (!ifs.read(reinterpret_cast<char*>(&ls.fc2_bias), sizeof(int32_t))) {
                return false;
            }
            if (!ifs.read(reinterpret_cast<char*>(ls.fc2_weights.data()),
                          static_cast<std::streamsize>(ls.fc2_weights.size() * sizeof(int8_t)))) {
                return false;
            }
        }

        if (ifs.good()) {
            network_loaded_ = true;
            return true;
        }
        return false;
    }

    // 导出符合 23,107,844 字节规范的二进制网络文件（用于本地自检验证 load_network）
    [[nodiscard]] bool save_network(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) return false;

        // 为精确对齐 23,107,844 字节：
        // FT 参数 = 1,024 + 23,068,672 = 23,069,696 字节
        // 4 个 LayerStack (含 4 个 FC_HASH 16字节 + 4 * 9,508 字节参数 = 38,048 字节)
        // Header (Version 4B + NetHash 4B + DescLen 4B + FT_HASH 4B = 16B) + DescLen(84B) = 100B
        // 总计: 23,069,696 + 38,048 + 100 = 23,107,844 字节
        const std::string desc(84, 'V');
        const uint32_t version  = kVersion;
        const uint32_t net_hash = kNetHash;
        const uint32_t desc_len = static_cast<uint32_t>(desc.size());
        const uint32_t ft_hash  = kFtHash;
        const uint32_t fc_hash  = kFcHash;

        ofs.write(reinterpret_cast<const char*>(&version),  sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&net_hash), sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&desc_len), sizeof(uint32_t));
        ofs.write(desc.data(), static_cast<std::streamsize>(desc.size()));
        ofs.write(reinterpret_cast<const char*>(&ft_hash),  sizeof(uint32_t));

        ofs.write(reinterpret_cast<const char*>(params_->ft_biases.data()),
                  static_cast<std::streamsize>(params_->ft_biases.size() * sizeof(int16_t)));
        ofs.write(reinterpret_cast<const char*>(params_->ft_weights.data()),
                  static_cast<std::streamsize>(params_->ft_weights.size() * sizeof(int8_t)));

        for (size_t b = 0; b < kNumLsBuckets; ++b) {
            const auto& ls = params_->buckets[b];
            ofs.write(reinterpret_cast<const char*>(&fc_hash), sizeof(uint32_t));
            ofs.write(reinterpret_cast<const char*>(ls.fc0_biases.data()),
                      static_cast<std::streamsize>(ls.fc0_biases.size() * sizeof(int32_t)));
            ofs.write(reinterpret_cast<const char*>(ls.fc0_weights.data()),
                      static_cast<std::streamsize>(ls.fc0_weights.size() * sizeof(int8_t)));
            ofs.write(reinterpret_cast<const char*>(ls.fc1_biases.data()),
                      static_cast<std::streamsize>(ls.fc1_biases.size() * sizeof(int32_t)));
            ofs.write(reinterpret_cast<const char*>(ls.fc1_weights.data()),
                      static_cast<std::streamsize>(ls.fc1_weights.size() * sizeof(int8_t)));
            ofs.write(reinterpret_cast<const char*>(&ls.fc2_bias), sizeof(int32_t));
            ofs.write(reinterpret_cast<const char*>(ls.fc2_weights.data()),
                      static_cast<std::streamsize>(ls.fc2_weights.size() * sizeof(int8_t)));
        }

        return ofs.good();
    }

    // ------------------------------------------------------------------------
    // 累加器全量刷新与增量更新 (make_move / undo_move 联动)
    // ------------------------------------------------------------------------

    // 对单个视角 (White 或 Black) 进行 O(N) 全量刷新（仅在初始化或该视角王移动时触发）
    void refresh_perspective(const Position& pos, Color perspective, Accumulator& acc) const noexcept {
        const Bitboard king_bb = pos.pieces(perspective, PieceType::King);
        if (king_bb.empty()) return;
        const Square ksq = king_bb.lsb();

        auto& vec = acc[perspective];
        vec = params_->ft_biases;

        for (Square sq : pos.occupancy()) {
            const Piece p = pos.piece_on(sq);
            const int feat_idx = halfkav2_index(perspective, ksq, p, sq);
            const int8_t* w_row = &params_->ft_weights[static_cast<size_t>(feat_idx) * kFtHalfDims];
            for (size_t i = 0; i < kFtHalfDims; ++i) {
                vec[i] = static_cast<int16_t>(vec[i] + static_cast<int16_t>(w_row[i]));
            }
        }

        if (perspective == Color::White) {
            acc.white_king_sq  = ksq;
            acc.computed_white = true;
        } else {
            acc.black_king_sq  = ksq;
            acc.computed_black = true;
        }
    }

    // 全量刷新双视角累加器
    void refresh_accumulator(const Position& pos, Accumulator& acc) const noexcept {
        refresh_perspective(pos, Color::White, acc);
        refresh_perspective(pos, Color::Black, acc);
    }

    // 在 make_move 后对 Accumulator 进行快速增量更新：
    // - 若某方王发生移动，则仅对该方视角执行 O(N) 全量刷新；
    // - 否则仅对变动的棋子权重从 int8_t 符号扩展并加减至 int16_t[512] 累加器。
    void update_on_make_move(
        const Position& pos_after_move,
        const DirtyFeatures& df,
        Accumulator& acc) const noexcept
    {
        // White 视角更新
        if (!acc.computed_white || df.white_king_moved) {
            refresh_perspective(pos_after_move, Color::White, acc);
        } else {
            apply_delta_perspective(Color::White, acc.white_king_sq, df, acc.white, false);
        }

        // Black 视角更新
        if (!acc.computed_black || df.black_king_moved) {
            refresh_perspective(pos_after_move, Color::Black, acc);
        } else {
            apply_delta_perspective(Color::Black, acc.black_king_sq, df, acc.black, false);
        }
    }

    // 在 undo_move 后对 Accumulator 进行快速逆向回滚（或直接恢复栈顶 Accumulator）
    void update_on_undo_move(
        const Position& pos_after_undo,
        const DirtyFeatures& df,
        Accumulator& acc) const noexcept
    {
        if (!acc.computed_white || df.white_king_moved) {
            refresh_perspective(pos_after_undo, Color::White, acc);
        } else {
            apply_delta_perspective(Color::White, acc.white_king_sq, df, acc.white, true);
        }

        if (!acc.computed_black || df.black_king_moved) {
            refresh_perspective(pos_after_undo, Color::Black, acc);
        } else {
            apply_delta_perspective(Color::Black, acc.black_king_sq, df, acc.black, true);
        }
    }

    // 便捷联动封装：一次完成 Position::make_move + Accumulator 增量更新
    void make_move_with_nnue(Position& pos, Move m, StateInfo& st, Accumulator& acc) const {
        const Color us    = pos.side_to_move();
        const Piece moved = pos.piece_on(m.from_sq());
        pos.make_move(m, st);
        const DirtyFeatures df = extract_dirty_features(us, moved, m, st);
        update_on_make_move(pos, df, acc);
    }

    // 便捷联动封装：一次完成 Position::undo_move + Accumulator 逆向回滚
    void undo_move_with_nnue(Position& pos, Move m, const StateInfo& st, Accumulator& acc) const noexcept {
        const Color us = ~pos.side_to_move();
        const Piece moved_before = m.is_promotion()
            ? make_piece(us, PieceType::Pawn)
            : pos.piece_on(m.to_sq());
        const DirtyFeatures df = extract_dirty_features(us, moved_before, m, st);
        pos.undo_move(m, st);
        update_on_undo_move(pos, df, acc);
    }

    // ------------------------------------------------------------------------
    // 前向推理接口 evaluate(pos, acc)
    // ------------------------------------------------------------------------
    [[nodiscard]] int evaluate(const Position& pos, Accumulator& acc) const noexcept {
        if (!acc.computed_white) refresh_perspective(pos, Color::White, acc);
        if (!acc.computed_black) refresh_perspective(pos, Color::Black, acc);

        const Color us   = pos.side_to_move();
        const Color them = ~us;

        const auto& us_acc   = acc[us];
        const auto& them_acc = acc[them];

        // 1. 特征变换器成对乘法与激活 (Pairwise Clipped-ReLU):
        // 每个视角截断到 [0, 255]，前半区(256维)与后半区(256维)相乘并右移 9 位：
        // out[i] = (clamp(acc[i], 0, 255) * clamp(acc[i + 256], 0, 255)) >> 9 (范围 [0, 127])
        // Us 与 Them 拼接得到 512 维输入 ft_out
        alignas(64) std::array<uint8_t, kFc0InDims> ft_out{};
#if defined(__AVX2__)
        const __m256i zero   = _mm256_setzero_si256();
        const __m256i max255 = _mm256_set1_epi16(255);

        for (size_t i = 0; i < kFtPairwiseOut; i += 16) {
            __m256i u0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&us_acc[i]));
            __m256i u1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&us_acc[i + kFtPairwiseOut]));
            u0 = _mm256_min_epi16(_mm256_max_epi16(u0, zero), max255);
            u1 = _mm256_min_epi16(_mm256_max_epi16(u1, zero), max255);
            __m256i prod_u = _mm256_srli_epi16(_mm256_mullo_epi16(u0, u1), 9);
            __m256i pack_u = _mm256_permute4x64_epi64(_mm256_packus_epi16(prod_u, prod_u), 0b00001000);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&ft_out[i]), _mm256_castsi256_si128(pack_u));

            __m256i t0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&them_acc[i]));
            __m256i t1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&them_acc[i + kFtPairwiseOut]));
            t0 = _mm256_min_epi16(_mm256_max_epi16(t0, zero), max255);
            t1 = _mm256_min_epi16(_mm256_max_epi16(t1, zero), max255);
            __m256i prod_t = _mm256_srli_epi16(_mm256_mullo_epi16(t0, t1), 9);
            __m256i pack_t = _mm256_permute4x64_epi64(_mm256_packus_epi16(prod_t, prod_t), 0b00001000);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&ft_out[i + kFtPairwiseOut]), _mm256_castsi256_si128(pack_t));
        }
#else
        for (size_t i = 0; i < kFtPairwiseOut; ++i) {
            const int32_t u0 = std::clamp<int32_t>(us_acc[i],                  0, 255);
            const int32_t u1 = std::clamp<int32_t>(us_acc[i + kFtPairwiseOut], 0, 255);
            ft_out[i] = static_cast<uint8_t>((u0 * u1) >> 9);

            const int32_t t0 = std::clamp<int32_t>(them_acc[i],                  0, 255);
            const int32_t t1 = std::clamp<int32_t>(them_acc[i + kFtPairwiseOut], 0, 255);
            ft_out[i + kFtPairwiseOut] = static_cast<uint8_t>((t0 * t1) >> 9);
        }
#endif

        // 2. 根据总棋子数动态选择 LayerStack 分桶: bucket = std::clamp((piece_count - 1) / 8, 0, 3)
        const int piece_count = pos.occupancy().popcount();
        const int bucket_idx  = select_layer_stack_bucket(piece_count);
        const auto& ls        = params_->buckets[static_cast<size_t>(bucket_idx)];

        // 3. fc_0 (512 -> 16): int8_t 权重 + int32_t 偏置，输出 >> 7 (因权重缩放为 128)
        alignas(64) std::array<int32_t, kFc0OutDims> fc0_raw{};
#if defined(__AVX2__)
        const __m256i ones = _mm256_set1_epi16(1);
        for (size_t o = 0; o < kFc0OutDims; ++o) {
            const int8_t* w_row = &ls.fc0_weights[o * kFc0InDims];
            __m256i sum32 = _mm256_setzero_si256();
            for (size_t i = 0; i < kFc0InDims; i += 32) {
                __m256i in8    = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ft_out[i]));
                __m256i wt8    = _mm256_load_si256(reinterpret_cast<const __m256i*>(&w_row[i]));
                __m256i prod16 = _mm256_maddubs_epi16(in8, wt8);
                __m256i prod32 = _mm256_madd_epi16(prod16, ones);
                sum32          = _mm256_add_epi32(sum32, prod32);
            }
            __m128i low128  = _mm256_castsi256_si128(sum32);
            __m128i high128 = _mm256_extracti128_si256(sum32, 1);
            __m128i sum128  = _mm_add_epi32(low128, high128);
            sum128          = _mm_hadd_epi32(sum128, sum128);
            sum128          = _mm_hadd_epi32(sum128, sum128);
            const int32_t total = _mm_cvtsi128_si32(sum128) + ls.fc0_biases[o];
            fc0_raw[o] = total >> kFc0WeightShift;
        }
#else
        for (size_t o = 0; o < kFc0OutDims; ++o) {
            int32_t sum = ls.fc0_biases[o];
            const int8_t* w_row = &ls.fc0_weights[o * kFc0InDims];
            for (size_t i = 0; i < kFc0InDims; ++i) {
                sum += static_cast<int32_t>(ft_out[i]) * static_cast<int32_t>(w_row[i]);
            }
            fc0_raw[o] = sum >> kFc0WeightShift;
        }
#endif

        // 提取 l1_skip = fc0_out[14] - fc0_out[15] (尺度为 128)
        const int32_t l1_skip = fc0_raw[14] - fc0_raw[15];

        // 双激活拼接 l1x_: [ (fc0_out^2) >> 7, fc0_out ] 得到 32 维输入 fc0_act
        alignas(64) std::array<uint8_t, kFc1InDims> fc0_act{};
        for (size_t i = 0; i < kFc0OutDims; ++i) {
            const int32_t clipped = std::clamp<int32_t>(fc0_raw[i], 0, 127);
            fc0_act[i]               = static_cast<uint8_t>((clipped * clipped) >> 7); // SqrCReLU
            fc0_act[i + kFc0OutDims] = static_cast<uint8_t>(clipped);                  // CReLU
        }

        // 4. fc_1 (32 -> 32): int8_t 权重 + int32_t 偏置，输出 >> 6 (因权重缩放为 64)
        alignas(64) std::array<int32_t, kFc1OutDims> fc1_raw{};
        for (size_t o = 0; o < kFc1OutDims; ++o) {
            int32_t sum = ls.fc1_biases[o];
            const int8_t* w_row = &ls.fc1_weights[o * kFc1InDims];
            for (size_t i = 0; i < kFc1InDims; ++i) {
                sum += static_cast<int32_t>(fc0_act[i]) * static_cast<int32_t>(w_row[i]);
            }
            fc1_raw[o] = sum >> kFc1WeightShift;
        }

        // 严格对标 layer_stacks.py: l3_input = torch.cat([l1x_, l2x_], dim=1)
        // 前 32 维 (0..31) 为 fc0_act (l1x_)，后 64 维 (32..95) 为 fc1_act (l2x_ = [(fc1_out^2) >> 7, fc1_out])
        alignas(64) std::array<uint8_t, kFc2InDims> fc2_in{};
        for (size_t i = 0; i < kFc1InDims; ++i) {
            fc2_in[i] = fc0_act[i]; // 0..31: fc0 双激活 (l1x_)
        }
        for (size_t i = 0; i < kFc1OutDims; ++i) {
            const int32_t clipped = std::clamp<int32_t>(fc1_raw[i], 0, 127);
            fc2_in[32 + i]               = static_cast<uint8_t>((clipped * clipped) >> 7); // 32..63: fc1 SqrCReLU
            fc2_in[32 + kFc1OutDims + i] = static_cast<uint8_t>(clipped);                  // 64..95: fc1 CReLU
        }

        // 5. fc_2 (96 -> 1): int8_t 权重 + int32_t 偏置 (fc2_sum 量化尺度为 128 * 128 = 16384)
        int32_t fc2_sum = ls.fc2_bias;
        for (size_t i = 0; i < kFc2InDims; ++i) {
            fc2_sum += static_cast<int32_t>(fc2_in[i]) * static_cast<int32_t>(ls.fc2_weights[i]);
        }

        // 合成对齐尺度 (16384): combined = fc2_sum + (l1_skip << 7)
        // 最终 centipawn 评分: cp = static_cast<int>((static_cast<int64_t>(combined) * 600) / 16384)
        const int32_t combined = fc2_sum + (l1_skip << kSkipScaleShift);
        int cp = static_cast<int>((static_cast<int64_t>(combined) * kNnue2ScoreInt) / kOutputScale);

        // 1. 全局残局缩放因子体系 (Endgame Scale Factor System - 对标 Stockfish)
        const int scale = calculate_endgame_scale_factor(pos);
        if (scale < 64) {
            cp = (cp * scale) / 64;
        }

        // 2. 50 回合渐近衰减 (Rule 50 Proximity Scaling)
        // 超过 60 半回合无兑子/推兵时，将优势平滑压缩向 0，防止死线边缘做自大幻觉晃荡
        const int hmc = pos.halfmove_clock();
        if (hmc >= 60 && std::abs(cp) < 30000) {
            const int r50_factor = std::clamp(100 - hmc, 0, 40);
            cp = (cp * r50_factor) / 40;
        }

        return cp;
    }

    [[nodiscard]] const NetworkParameters& parameters() const noexcept { return *params_; }

public:
    [[nodiscard]] bool is_network_loaded() const noexcept { return network_loaded_; }

private:
    std::unique_ptr<NetworkParameters> params_;
    bool                               network_loaded_ = false;

    void apply_delta_perspective(
        Color perspective,
        Square ksq,
        const DirtyFeatures& df,
        std::array<int16_t, kFtHalfDims>& acc_vec,
        bool reverse) const noexcept
    {
        // 正向 make_move: 减去 removed，加上 added
        // 逆向 undo_move: 加上 removed，减去 added
        for (size_t r = 0; r < df.num_removed; ++r) {
            const int feat_idx = halfkav2_index(perspective, ksq, df.removed[r].piece, df.removed[r].sq);
            const int8_t* w_row = &params_->ft_weights[static_cast<size_t>(feat_idx) * kFtHalfDims];
#if defined(__AVX2__)
            for (size_t i = 0; i < kFtHalfDims; i += 16) {
                __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&w_row[i]));
                __m256i w16 = _mm256_cvtepi8_epi16(w8);
                __m256i a16 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc_vec[i]));
                a16 = (!reverse) ? _mm256_sub_epi16(a16, w16) : _mm256_add_epi16(a16, w16);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&acc_vec[i]), a16);
            }
#else
            if (!reverse) {
                for (size_t i = 0; i < kFtHalfDims; ++i) {
                    acc_vec[i] = static_cast<int16_t>(acc_vec[i] - static_cast<int16_t>(w_row[i]));
                }
            } else {
                for (size_t i = 0; i < kFtHalfDims; ++i) {
                    acc_vec[i] = static_cast<int16_t>(acc_vec[i] + static_cast<int16_t>(w_row[i]));
                }
            }
#endif
        }

        for (size_t a = 0; a < df.num_added; ++a) {
            const int feat_idx = halfkav2_index(perspective, ksq, df.added[a].piece, df.added[a].sq);
            const int8_t* w_row = &params_->ft_weights[static_cast<size_t>(feat_idx) * kFtHalfDims];
#if defined(__AVX2__)
            for (size_t i = 0; i < kFtHalfDims; i += 16) {
                __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&w_row[i]));
                __m256i w16 = _mm256_cvtepi8_epi16(w8);
                __m256i a16 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc_vec[i]));
                a16 = (!reverse) ? _mm256_add_epi16(a16, w16) : _mm256_sub_epi16(a16, w16);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&acc_vec[i]), a16);
            }
#else
            if (!reverse) {
                for (size_t i = 0; i < kFtHalfDims; ++i) {
                    acc_vec[i] = static_cast<int16_t>(acc_vec[i] + static_cast<int16_t>(w_row[i]));
                }
            } else {
                for (size_t i = 0; i < kFtHalfDims; ++i) {
                    acc_vec[i] = static_cast<int16_t>(acc_vec[i] - static_cast<int16_t>(w_row[i]));
                }
            }
#endif
        }
    }
};

} // namespace valkyrie::nnue

// ============================================================================
// 5. 阶段三 NNUE 自检与测试入口 (当定义 VALKYRIE_STAGE3_MAIN 时编译为独立可执行文件)
//    g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -Werror -x c++ -DVALKYRIE_STAGE3_MAIN nnue.hpp -o nnue_test
// ============================================================================
#ifdef VALKYRIE_STAGE3_MAIN
int main(int argc, char* argv[]) {
    using namespace valkyrie;

    static_assert((nnue::kFeatureHash ^ (512U * 2U)) == nnue::kFtHash, "FT_HASH mismatch");
    static_assert((nnue::kFtHash ^ nnue::kFcHash) == nnue::kNetHash,   "NET_HASH mismatch");

    std::cout << "=== Valkyrie C++20 Engine (nnue.hpp) - Stage 3 HalfKAv2 512x2 NNUE Self-Test ===\n";
    std::cout << "VERSION      : 0x" << std::hex << nnue::kVersion     << "\n";
    std::cout << "FEATURE_HASH : 0x" << std::hex << nnue::kFeatureHash << "\n";
    std::cout << "FT_HASH      : 0x" << std::hex << nnue::kFtHash      << "\n";
    std::cout << "FC_HASH      : 0x" << std::hex << nnue::kFcHash      << "\n";
    std::cout << "NET_HASH     : 0x" << std::hex << nnue::kNetHash     << std::dec << "\n";
    std::cout << "Target Bytes : " << nnue::kExpectedFileSize << " bytes (45,056 x 512 int8 FT)\n";

    nnue::Evaluator evaluator;

    // 若命令行传入了正式 .nnue 网络路径则直接加载；否则生成临时 23,107,844 字节网络文件并测试完整二进制加载流程
    const std::string net_path = (argc > 1) ? argv[1] : "/tmp/valkyrie_test_net.nnue";
    if (argc <= 1) {
        const bool saved = evaluator.save_network(net_path);
        assert(saved);
    }

    const bool loaded = evaluator.load_network(net_path);
    std::cout << "load_network(\"" << net_path << "\") : " << (loaded ? "SUCCESS" : "FAILED") << "\n";
    assert(loaded);

    Position pos;
    const bool fen_ok = pos.set_from_fen(Position::kStartFEN);
    assert(fen_ok);

    nnue::Accumulator acc{};
    evaluator.refresh_accumulator(pos, acc);

    const int start_cp = evaluator.evaluate(pos, acc);
    std::cout << "Startpos Static Eval : " << start_cp << " cp (LayerStack Bucket = "
              << nnue::select_layer_stack_bucket(pos.occupancy().popcount()) << ")\n";

    // 验证增量更新 make_move_with_nnue / undo_move_with_nnue 与全量刷新 refresh_accumulator 100% 一致
    StateInfo st1{}, st2{};
    const Move m1(Square::e2, Square::e4, MoveFlag::DoublePawnPush);
    const Move m2(Square::e7, Square::e5, MoveFlag::DoublePawnPush);

    evaluator.make_move_with_nnue(pos, m1, st1, acc);
    evaluator.make_move_with_nnue(pos, m2, st2, acc);

    nnue::Accumulator scratch_acc{};
    evaluator.refresh_accumulator(pos, scratch_acc);
    assert(acc.white == scratch_acc.white);
    assert(acc.black == scratch_acc.black);
    std::cout << "After 1. e4 e5 Eval  : " << evaluator.evaluate(pos, acc)
              << " cp (Incremental == Scratch Acc: VERIFIED)\n";

    evaluator.undo_move_with_nnue(pos, m2, st2, acc);
    evaluator.undo_move_with_nnue(pos, m1, st1, acc);
    assert(evaluator.evaluate(pos, acc) == start_cp);
    std::cout << "After Undo Eval      : " << evaluator.evaluate(pos, acc) << " cp (Restored 100%)\n";

    return 0;
}
#endif
