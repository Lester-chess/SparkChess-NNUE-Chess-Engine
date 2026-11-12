#include <functional>
/**
 * @file search.hpp
 * @brief Sparkfish C++20 Chess Engine - Stage 4: Transposition Table (TT),
 *        Staged Move Ordering (TT / MVV-LVA + SEE / Killers / Countermove / History),
 *        Quiescence Search, Iterative Deepening + Aspiration Windows, and
 *        PVS with RFP, NMP, LMR, LMP, and Singular Extensions.
 * @standard C++20 (-std=c++20 / /std:c++20)
 * @license MIT (100% Original Architecture designed for HalfKAv2 512x2 NNUE)
 */

#pragma once

#include "position.hpp"
#include "movegen.hpp"
#include "nnue.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>

namespace valkyrie::search {

// ============================================================================
// 1. 搜索常量与杀棋分数 (Mate Score <-> Ply 标准转化)
// ============================================================================

inline constexpr int kMaxPly        = 128;
inline constexpr int kInfiniteScore = 32000;
inline constexpr int kMateScore     = 31000;
inline constexpr int kMateThreshold = kMateScore - kMaxPly; // 30872
inline constexpr int kValueNone     = 32500;
inline constexpr int kDrawScore     = 0;

// 将搜索树内的杀棋分转化为置换表存储分（以当前节点距离将死的步数为准，消除根距 Ply 偏移）
[[nodiscard]] constexpr int score_to_tt(int score, int ply) noexcept {
    if (score >= kMateThreshold)  return score + ply;
    if (score <= -kMateThreshold) return score - ply;
    return score;
}

// 将置换表读出的杀棋分还原为当前搜索树 Ply 下的杀棋分
[[nodiscard]] constexpr int score_from_tt(int score, int ply) noexcept {
    if (score >= kMateThreshold)  return score - ply;
    if (score <= -kMateThreshold) return score + ply;
    return score;
}

[[nodiscard]] constexpr bool is_mate_score(int score) noexcept {
    return score >= kMateThreshold || score <= -kMateThreshold;
}

[[nodiscard]] constexpr int cp_piece_value(PieceType pt) noexcept {
    switch (pt) {
        case PieceType::Pawn:   return 100;
        case PieceType::Knight: return 320;
        case PieceType::Bishop: return 330;
        case PieceType::Rook:   return 500;
        case PieceType::Queen:  return 950;
        default: return 0;
    }
}

[[nodiscard]] constexpr int cp_piece_value(Piece p) noexcept {
    return (p == Piece::None) ? 0 : cp_piece_value(piece_type(p));
}

// 子力基础价值表（用于 MVV-LVA、SEE 静态交换评估与 Delta Pruning）
inline constexpr std::array<int, 7> kPieceValue = {
    208,   // Pawn (匹配 NNUE 内部单位)
    781,   // Knight
    825,   // Bishop
    1276,  // Rook
    2538,  // Queen
    30000, // King
    0      // None
};

[[nodiscard]] constexpr int piece_value(PieceType pt) noexcept {
    return kPieceValue[static_cast<size_t>(pt)];
}

[[nodiscard]] constexpr int piece_value(Piece p) noexcept {
    return (p == Piece::None) ? 0 : piece_value(piece_type(p));
}

// 判断某方是否拥有除王和兵以外的子力（用于 NMP 残局 Zugzwang 过滤）
[[nodiscard]] inline bool has_non_pawn_material(const Position& pos, Color side) noexcept {
    const Bitboard non_pawn =
        pos.pieces(side, PieceType::Knight) |
        pos.pieces(side, PieceType::Bishop) |
        pos.pieces(side, PieceType::Rook)   |
        pos.pieces(side, PieceType::Queen);
    return non_pawn.any();
}

// ▼▼▼ 在这里直接粘贴下面这个函数即可！▼▼▼
[[nodiscard]] inline bool is_passed_pawn(const Position& pos, Square sq, Color c) noexcept {
    const int f = static_cast<int>(file_of(sq));
    const int r = static_cast<int>(rank_of(sq));
    uint64_t files = (1ULL << f);
    if (f > 0) files |= (1ULL << (f - 1));
    if (f < 7) files |= (1ULL << (f + 1));
    uint64_t span = 0ULL;
    for (int rank = 0; rank < 8; ++rank) span |= (files << (rank * 8));
    const uint64_t forward_mask = (c == Color::White)
        ? ~((1ULL << ((r + 1) * 8)) - 1ULL)
        : ((1ULL << (r * 8)) - 1ULL);
    return (pos.pieces(~c, PieceType::Pawn).raw() & span & forward_mask) == 0ULL;
}
// ▲▲▲ 添加结束 ▲▲▲

// 计算纯兵型 Zobrist 键（用于 CorrHist 评估校准历史）
[[nodiscard]] inline uint64_t compute_pawn_key(const Position& pos) noexcept {
    uint64_t k = 0ULL;
    for (Square s : pos.pieces(Color::White, PieceType::Pawn)) {
        k ^= zobrist::kKeys.piece_sq[0][sq_index(s)];
    }
    for (Square s : pos.pieces(Color::Black, PieceType::Pawn)) {
        k ^= zobrist::kKeys.piece_sq[6][sq_index(s)];
    }
    return k;
}

// 计算单方非兵子力（马、象、车、后、王）Zobrist 键
[[nodiscard]] inline uint64_t compute_non_pawn_key(const Position& pos, Color side) noexcept {
    uint64_t k = 0ULL;
    const int base = (side == Color::White) ? 0 : 6;
    for (int pt = 1; pt <= 5; ++pt) {
        for (Square s : pos.pieces(side, static_cast<PieceType>(pt))) {
            k ^= zobrist::kKeys.piece_sq[base + pt][sq_index(s)];
        }
    }
    return k;
}

// 快速估算下一步走完后的 Zobrist Key（用于向 CPU 提前发出 TT 缓存预取指令）
[[nodiscard]] inline uint64_t approximate_key_after(const Position& pos, Move m) noexcept {
    uint64_t k = pos.zobrist_key() ^ zobrist::kKeys.side_to_move;
    const Square from = m.from_sq();
    const Square to = m.to_sq();
    const Piece moved = pos.piece_on(from);
    if (moved != Piece::None) {
        k ^= zobrist::kKeys.piece_sq[static_cast<size_t>(moved)][sq_index(from)];
        const Piece dest_p = m.is_promotion() ? make_piece(pos.side_to_move(), m.promotion_type()) : moved;
        k ^= zobrist::kKeys.piece_sq[static_cast<size_t>(dest_p)][sq_index(to)];
    }
    const Piece cap = pos.piece_on(to);
    if (cap != Piece::None) {
        k ^= zobrist::kKeys.piece_sq[static_cast<size_t>(cap)][sq_index(to)];
    }
    return k;
}
// ============================================================================
// 2. 置换表 (Transposition Table, TT) - 16 字节紧凑对齐 Entry 与动态 MB 分配
// ============================================================================

enum class TTBound : uint8_t {
    None       = 0,
    UpperBound = 1, // Alpha (Fail-Low)
    LowerBound = 2, // Beta  (Fail-High)
    Exact      = 3  // PV Exact
};

// 紧凑 16 字节 TTEntry（每个 64 字节缓存行恰好容纳 4 个条目）
struct alignas(16) TTEntry {
    uint64_t key         = 0ULL; // 完整 64-bit Zobrist Key 防止高并发/深层伪碰撞
    uint16_t best_move   = 0;    // Move::raw()
    int16_t  score       = 0;    // 经过 score_to_tt 调整后的分数
    int16_t  static_eval = 0;    // 静态 NNUE 评估缓存
    int8_t   depth       = -1;   // 搜索深度
    uint8_t  gen_bound   = 0;    // 高 6 位存 age (0..63)，低 2 位存 TTBound

    [[nodiscard]] constexpr TTBound bound() const noexcept {
        return static_cast<TTBound>(gen_bound & 0x03U);
    }

    [[nodiscard]] constexpr uint8_t age() const noexcept {
        return static_cast<uint8_t>(gen_bound >> 2U);
    }

    [[nodiscard]] constexpr Move move() const noexcept {
        return Move(best_move);
    }
};

static_assert(sizeof(TTEntry) == 16, "TTEntry must be exactly 16 bytes for cache-line alignment");

// 64 字节硬件缓存行对齐的 4 路组相联簇 (4-way Set-Associative Cluster)
struct alignas(64) TTCluster {
    std::array<TTEntry, 4> entries{};
};

static_assert(sizeof(TTCluster) == 64, "TTCluster must match 64-byte hardware cache line");

class TranspositionTable {
public:
    explicit TranspositionTable(size_t size_mb = 16) {
        resize_mb(size_mb);
    }

    void resize_mb(size_t size_mb) {
        size_mb = std::clamp<size_t>(size_mb, 1, 4096);
        const size_t target_bytes = size_mb * 1024ULL * 1024ULL;
        const size_t max_clusters = target_bytes / sizeof(TTCluster);

        size_t pow2 = 1ULL;
        while ((pow2 << 1U) <= max_clusters) {
            pow2 <<= 1U;
        }

        table_.assign(pow2, TTCluster{});
        mask_ = pow2 - 1ULL;
        age_  = 0;
    }

    void clear() noexcept {
        std::fill(table_.begin(), table_.end(), TTCluster{});
        age_ = 0;
    }

    void new_search() noexcept {
        age_ = static_cast<uint8_t>((age_ + 1U) & 0x3FU);
    }

    [[nodiscard]] size_t size_entries() const noexcept { return table_.size() * 4; }
    [[nodiscard]] size_t size_mb() const noexcept {
        return (table_.size() * sizeof(TTCluster)) / (1024ULL * 1024ULL);
    }

    [[nodiscard]] bool probe(uint64_t key, TTEntry& out_entry) const noexcept {
        if (table_.empty()) return false;
        const TTCluster& c = table_[key & mask_];
        for (size_t i = 0; i < 4; ++i) {
            if (c.entries[i].key == key && c.entries[i].bound() != TTBound::None) {
                out_entry = c.entries[i];
                return true;
            }
        }
        return false;
    }

    // 硬件预取：提前将对应 key 的 64 字节 TTCluster 载入 L1 缓存行
    void prefetch(uint64_t key) const noexcept {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&table_[key & mask_]);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        _mm_prefetch(reinterpret_cast<const char*>(&table_[key & mask_]), _MM_HINT_T0);
#endif
    }

    void store(
        uint64_t key,
        Move best_move,
        int score,
        int static_eval,
        int depth,
        TTBound bound,
        int ply) noexcept
    {
        if (table_.empty()) return;
        TTCluster& c = table_[key & mask_];

        // 1. 同局面命中更新
        for (size_t i = 0; i < 4; ++i) {
            if (c.entries[i].key == key && c.entries[i].bound() != TTBound::None) {
                if (best_move.is_none()) {
                    best_move = c.entries[i].move();
                }
                if (bound == TTBound::Exact ||
                    depth + 2 >= static_cast<int>(c.entries[i].depth) ||
                    c.entries[i].age() != age_)
                {
                    c.entries[i].best_move   = best_move.raw();
                    c.entries[i].score       = static_cast<int16_t>(std::clamp(score_to_tt(score, ply), -32767, 32767));
                    c.entries[i].static_eval = static_cast<int16_t>(std::clamp(static_eval, -32767, 32767));
                    c.entries[i].depth       = static_cast<int8_t>(std::clamp(depth, 0, 127));
                    c.entries[i].gen_bound   = static_cast<uint8_t>((age_ << 2U) | static_cast<uint8_t>(bound));
                }
                return;
            }
        }

        // 2. 空槽填入
        for (size_t i = 0; i < 4; ++i) {
            if (c.entries[i].bound() == TTBound::None) {
                c.entries[i].key         = key;
                c.entries[i].best_move   = best_move.raw();
                c.entries[i].score       = static_cast<int16_t>(std::clamp(score_to_tt(score, ply), -32767, 32767));
                c.entries[i].static_eval = static_cast<int16_t>(std::clamp(static_eval, -32767, 32767));
                c.entries[i].depth       = static_cast<int8_t>(std::clamp(depth, 0, 127));
                c.entries[i].gen_bound   = static_cast<uint8_t>((age_ << 2U) | static_cast<uint8_t>(bound));
                return;
            }
        }

        // 3. 淘汰策略：挑选保护分最低（最老、最浅）的槽位
        size_t victim_idx = 0;
        int min_val = 999999;
        for (size_t i = 0; i < 4; ++i) {
            int val = static_cast<int>(c.entries[i].depth) * 2;
            if (c.entries[i].age() == age_) val += 128;
            if (c.entries[i].bound() == TTBound::Exact) val += 16;
            if (val < min_val) {
                min_val = val;
                victim_idx = i;
            }
        }

        // 严格保护当前轮次的深层条目免遭叶子节点 (QSearch depth=0) 覆盖
        if (depth == 0 && c.entries[victim_idx].age() == age_ && c.entries[victim_idx].depth > 0) {
            return;
        }

        c.entries[victim_idx].key         = key;
        c.entries[victim_idx].best_move   = best_move.raw();
        c.entries[victim_idx].score       = static_cast<int16_t>(std::clamp(score_to_tt(score, ply), -32767, 32767));
        c.entries[victim_idx].static_eval = static_cast<int16_t>(std::clamp(static_eval, -32767, 32767));
        c.entries[victim_idx].depth       = static_cast<int8_t>(std::clamp(depth, 0, 127));
        c.entries[victim_idx].gen_bound   = static_cast<uint8_t>((age_ << 2U) | static_cast<uint8_t>(bound));
    }

    [[nodiscard]] int hashfull() const noexcept {
        if (table_.empty()) return 0;
        const size_t sample_clusters = std::min<size_t>(table_.size(), 250);
        int used = 0;
        for (size_t i = 0; i < sample_clusters; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                if (table_[i].entries[j].bound() != TTBound::None && table_[i].entries[j].age() == age_) {
                    ++used;
                }
            }
        }
        return (used * 1000) / static_cast<int>(sample_clusters * 4);
    }

private:
    std::vector<TTCluster> table_{};
    size_t                 mask_ = 0;
    uint8_t                age_  = 0;
};

// ============================================================================
// 3. 静态交换评估 (Static Exchange Evaluation, SEE) - 用于区分 Good / Bad Captures
// ============================================================================

// 计算盘面各方攻击目标格 sq 的所有攻击子位棋盘（给定自定义占位 occ）
[[nodiscard]] inline Bitboard attackers_to_sq(const Position& pos, Square sq, Bitboard occ) noexcept {
    const Bitboard w_pawns   = pos.pieces(Color::White, PieceType::Pawn);
    const Bitboard b_pawns   = pos.pieces(Color::Black, PieceType::Pawn);
    const Bitboard knights   = pos.pieces(Color::White, PieceType::Knight) | pos.pieces(Color::Black, PieceType::Knight);
    const Bitboard kings     = pos.pieces(Color::White, PieceType::King)   | pos.pieces(Color::Black, PieceType::King);
    const Bitboard bishops_q = pos.pieces(Color::White, PieceType::Bishop) | pos.pieces(Color::Black, PieceType::Bishop) |
                               pos.pieces(Color::White, PieceType::Queen)  | pos.pieces(Color::Black, PieceType::Queen);
    const Bitboard rooks_q   = pos.pieces(Color::White, PieceType::Rook)   | pos.pieces(Color::Black, PieceType::Rook)   |
                               pos.pieces(Color::White, PieceType::Queen)  | pos.pieces(Color::Black, PieceType::Queen);

    return (attacks::pawn_attacks(Color::Black, sq) & w_pawns) |
           (attacks::pawn_attacks(Color::White, sq) & b_pawns) |
           (attacks::knight_attacks(sq)             & knights) |
           (attacks::king_attacks(sq)               & kings)   |
           (attacks::bishop_attacks(sq, occ)        & bishops_q) |
           (attacks::rook_attacks(sq, occ)          & rooks_q);
}

// 判断走法 m 的静态交换得分是否 >= threshold (O(1) 快速位棋盘射线揭露算法)
[[nodiscard]] inline bool see_ge(const Position& pos, Move m, int threshold = 0) noexcept {
    if (m.is_castling()) return threshold <= 0;

    const Square from = m.from_sq();
    const Square to   = m.to_sq();

    const Piece attacker_piece = pos.piece_on(from);
    if (attacker_piece == Piece::None) return false;

    int victim_val = 0;
    if (m.is_en_passant()) {
        victim_val = piece_value(PieceType::Pawn);
    } else {
        victim_val = piece_value(pos.piece_on(to));
    }

    if (m.is_promotion()) {
        victim_val += piece_value(m.promotion_type()) - piece_value(PieceType::Pawn);
    }

    int swap = victim_val - threshold;
    if (swap < 0) return false;

    const PieceType next_victim = m.is_promotion() ? m.promotion_type() : piece_type(attacker_piece);
    swap = piece_value(next_victim) - swap;
    if (swap <= 0) return true;

    Bitboard occ = pos.occupancy();
    occ.clear(from);
    if (m.is_en_passant()) {
        const Square ep_cap = (pos.side_to_move() == Color::White)
            ? static_cast<Square>(sq_index(to) - 8U)
            : static_cast<Square>(sq_index(to) + 8U);
        occ.clear(ep_cap);
    }
    occ.set(to);

    Bitboard attackers = attackers_to_sq(pos, to, occ) & occ;
    Color stm = ~pos.side_to_move();
    int res = 1;

    const Bitboard diag_sliders =
        pos.pieces(Color::White, PieceType::Bishop) | pos.pieces(Color::Black, PieceType::Bishop) |
        pos.pieces(Color::White, PieceType::Queen)  | pos.pieces(Color::Black, PieceType::Queen);
    const Bitboard orth_sliders =
        pos.pieces(Color::White, PieceType::Rook)   | pos.pieces(Color::Black, PieceType::Rook)   |
        pos.pieces(Color::White, PieceType::Queen)  | pos.pieces(Color::Black, PieceType::Queen);

    while (true) {
        const Bitboard stm_attackers = attackers & pos.pieces(stm);
        if (stm_attackers.empty()) break;

        res ^= 1;

        // 寻找最小价值攻击子 (Least Valuable Attacker)
        PieceType lva_pt = PieceType::None;
        Square    lva_sq = Square::None;
        for (int pt_i = 0; pt_i <= 5; ++pt_i) {
            const auto pt = static_cast<PieceType>(pt_i);
            const Bitboard sub = stm_attackers & pos.pieces(stm, pt);
            if (sub.any()) {
                lva_pt = pt;
                lva_sq = sub.lsb();
                break;
            }
        }

        if (lva_pt == PieceType::King) {
            // 若对方还有攻击子，则王吃子后会被反杀
            if ((attackers & pos.pieces(~stm)).any()) {
                res ^= 1;
            }
            break;
        }

        occ.clear(lva_sq);

        // 揭露后方X射线滑动攻击子
        if (lva_pt == PieceType::Pawn || lva_pt == PieceType::Bishop || lva_pt == PieceType::Queen) {
            attackers |= (attacks::bishop_attacks(to, occ) & diag_sliders);
        }
        if (lva_pt == PieceType::Rook || lva_pt == PieceType::Queen) {
            attackers |= (attacks::rook_attacks(to, occ) & orth_sliders);
        }
        attackers &= occ;

        swap = piece_value(lva_pt) - swap;
        if (swap < res) break;

        stm = ~stm;
    }

    return res != 0;
}

// ============================================================================
// 4. 历史启发式表与杀手/反制走法表 (Heuristics Table)
// ============================================================================

struct SearchHeuristics {
    static constexpr int kMaxHistory = 16384;

    // killers[ply][0..1]
    std::array<std::array<Move, 2>, kMaxPly> killers{};
    // countermoves[piece_12][to_64]
    std::array<std::array<Move, 64>, 13> countermoves{};
    // main_history[color_2][from_64][to_64]
    std::array<std::array<std::array<int32_t, 64>, 64>, 2> history{};
    // conthist: ply-1 (对手上一轮走的棋)
    std::array<std::array<std::array<std::array<int16_t, 64>, 13>, 64>, 13> conthist{};
    // conthist2: ply-2 (我方自身上一步走法形成的战略延续)
    std::array<std::array<std::array<std::array<int16_t, 64>, 13>, 64>, 13> conthist2{};
    // capture_history[moved_piece_12][to_64][captured_type_6]
    std::array<std::array<std::array<int16_t, 6>, 64>, 12> capture_history{};
    // correction_history[color_2][16384] (兵型 + 白/黑非兵子力多源评估校准历史)
    static constexpr size_t kCorrHistSize = 16384;
    std::array<std::array<int16_t, kCorrHistSize>, 2> corr_hist{};
    std::array<std::array<int16_t, kCorrHistSize>, 2> non_pawn_corr_w{};
    std::array<std::array<int16_t, kCorrHistSize>, 2> non_pawn_corr_b{};

    void clear() noexcept {
        for (auto& ch : corr_hist) ch.fill(0);
        for (auto& ch : non_pawn_corr_w) ch.fill(0);
        for (auto& ch : non_pawn_corr_b) ch.fill(0);
        for (auto& k : killers) k = { Move::null(), Move::null() };
        for (auto& cm : countermoves) cm.fill(Move::null());
        for (auto& side_h : history) {
            for (auto& from_h : side_h) {
                from_h.fill(0);
            }
        }
        for (auto& pp : conthist) {
            for (auto& ps : pp) {
                for (auto& mp : ps) {
                    mp.fill(0);
                }
            }
        }
        for (auto& pp : conthist2) {
            for (auto& ps : pp) {
                for (auto& mp : ps) {
                    mp.fill(0);
                }
            }
        }
        for (auto& p : capture_history) {
            for (auto& s : p) {
                s.fill(0);
            }
        }
    }

    // 衰减历史表
    void age_history() noexcept {
        for (auto& side_h : history) {
            for (auto& from_h : side_h) {
                for (auto& val : from_h) {
                    val /= 2;
                }
            }
        }
        for (auto& pp : conthist) {
            for (auto& ps : pp) {
                for (auto& mp : ps) {
                    for (auto& val : mp) {
                        val /= 2;
                    }
                }
            }
        }
        for (auto& pp : conthist2) {
            for (auto& ps : pp) {
                for (auto& mp : ps) {
                    for (auto& val : mp) {
                        val /= 2;
                    }
                }
            }
        }
        // 补齐吃子历史表衰减，防止中残局被开局已失效的吃子偏好误导
        for (auto& p : capture_history) {
            for (auto& s : p) {
                for (auto& val : s) {
                    val /= 2;
                }
            }
        }
    }

    // 1-ply 延续历史操作
    void update_conthist(Piece prev_p, Square prev_sq, Piece moved_p, Square to_sq, int bonus) noexcept {
        const auto pp = static_cast<size_t>(prev_p);
        const auto ps = static_cast<size_t>(sq_index(prev_sq));
        const auto mp = static_cast<size_t>(moved_p);
        const auto ms = static_cast<size_t>(sq_index(to_sq));
        if (pp < 12U && ps < 64U && mp < 12U && ms < 64U) {
            int16_t& entry = conthist[pp][ps][mp][ms];
            const int clamped = std::clamp(bonus, -kMaxHistory, kMaxHistory);
            entry += static_cast<int16_t>(clamped - (entry * std::abs(clamped)) / kMaxHistory);
        }
    }

    [[nodiscard]] int32_t get_conthist(Piece prev_p, Square prev_sq, Piece moved_p, Square to_sq) const noexcept {
        const auto pp = static_cast<size_t>(prev_p);
        const auto ps = static_cast<size_t>(sq_index(prev_sq));
        const auto mp = static_cast<size_t>(moved_p);
        const auto ms = static_cast<size_t>(sq_index(to_sq));
        if (pp < 12U && ps < 64U && mp < 12U && ms < 64U) {
            return conthist[pp][ps][mp][ms];
        }
        return 0;
    }

    // 2-ply 延续历史操作
    void update_conthist2(Piece prev_p, Square prev_sq, Piece moved_p, Square to_sq, int bonus) noexcept {
        const auto pp = static_cast<size_t>(prev_p);
        const auto ps = static_cast<size_t>(sq_index(prev_sq));
        const auto mp = static_cast<size_t>(moved_p);
        const auto ms = static_cast<size_t>(sq_index(to_sq));
        if (pp < 12U && ps < 64U && mp < 12U && ms < 64U) {
            int16_t& entry = conthist2[pp][ps][mp][ms];
            const int clamped = std::clamp(bonus, -kMaxHistory, kMaxHistory);
            entry += static_cast<int16_t>(clamped - (entry * std::abs(clamped)) / kMaxHistory);
        }
    }

    [[nodiscard]] int32_t get_conthist2(Piece prev_p, Square prev_sq, Piece moved_p, Square to_sq) const noexcept {
        const auto pp = static_cast<size_t>(prev_p);
        const auto ps = static_cast<size_t>(sq_index(prev_sq));
        const auto mp = static_cast<size_t>(moved_p);
        const auto ms = static_cast<size_t>(sq_index(to_sq));
        if (pp < 12U && ps < 64U && mp < 12U && ms < 64U) {
            return conthist2[pp][ps][mp][ms];
        }
        return 0;
    }

    void update_capture_history(Piece moved, Square to, PieceType captured, int bonus) noexcept {
        const auto p_idx = static_cast<size_t>(moved);
        const auto s_idx = static_cast<size_t>(sq_index(to));
        const auto c_idx = static_cast<size_t>(captured);
        if (p_idx < 12U && s_idx < 64U && c_idx < 6U) {
            int16_t& entry = capture_history[p_idx][s_idx][c_idx];
            const int clamped = std::clamp(bonus, -kMaxHistory, kMaxHistory);
            entry += static_cast<int16_t>(clamped - (entry * std::abs(clamped)) / kMaxHistory);
        }
    }

    [[nodiscard]] int32_t get_capture_history(Piece moved, Square to, PieceType captured) const noexcept {
        const auto p_idx = static_cast<size_t>(moved);
        const auto s_idx = static_cast<size_t>(sq_index(to));
        const auto c_idx = static_cast<size_t>(captured);
        if (p_idx < 12U && s_idx < 64U && c_idx < 6U) {
            return capture_history[p_idx][s_idx][c_idx];
        }
        return 0;
    }

    void update_killer(int ply, Move m) noexcept {
        if (ply < 0 || ply >= kMaxPly) return;
        if (killers[static_cast<size_t>(ply)][0] != m) {
            killers[static_cast<size_t>(ply)][1] = killers[static_cast<size_t>(ply)][0];
            killers[static_cast<size_t>(ply)][0] = m;
        }
    }

    void update_countermove(Piece prev_piece, Square prev_to, Move m) noexcept {
        const auto p_idx = static_cast<size_t>(prev_piece);
        const auto s_idx = static_cast<size_t>(sq_index(prev_to));
        if (p_idx < 12U && s_idx < 64U) {
            countermoves[p_idx][s_idx] = m;
        }
    }

    [[nodiscard]] Move get_countermove(Piece prev_piece, Square prev_to) const noexcept {
        const auto p_idx = static_cast<size_t>(prev_piece);
        const auto s_idx = static_cast<size_t>(sq_index(prev_to));
        if (p_idx < 12U && s_idx < 64U) {
            return countermoves[p_idx][s_idx];
        }
        return Move::null();
    }

    void update_history(Color c, Move m, int bonus) noexcept {
        const int clamped_bonus = std::clamp(bonus, -kMaxHistory, kMaxHistory);
        int32_t& entry = history[static_cast<size_t>(c)][sq_index(m.from_sq())][sq_index(m.to_sq())];
        entry += clamped_bonus - (entry * std::abs(clamped_bonus)) / kMaxHistory;
    }

    [[nodiscard]] int32_t get_history(Color c, Move m) const noexcept {
        return history[static_cast<size_t>(c)][sq_index(m.from_sq())][sq_index(m.to_sq())];
    }

    [[nodiscard]] int get_correction(Color c, uint64_t pawn_key, uint64_t np_w_key, uint64_t np_b_key) const noexcept {
        const size_t side = static_cast<size_t>(c);
        const int p = static_cast<int>(corr_hist[side][pawn_key & (kCorrHistSize - 1)]);
        const int nw = static_cast<int>(non_pawn_corr_w[side][np_w_key & (kCorrHistSize - 1)]);
        const int nb = static_cast<int>(non_pawn_corr_b[side][np_b_key & (kCorrHistSize - 1)]);
        return (p + nw + nb) / 24;
    }

    void update_correction(Color c, uint64_t pawn_key, uint64_t np_w_key, uint64_t np_b_key, int depth, int bonus) noexcept {
        const size_t side = static_cast<size_t>(c);
        const int scaled = std::clamp(bonus * depth, -2048, 2048);
        auto update_entry = [&](std::array<std::array<int16_t, kCorrHistSize>, 2>& table, uint64_t key) noexcept {
            int16_t& val = table[side][key & (kCorrHistSize - 1)];
            val += static_cast<int16_t>(scaled - (val * std::abs(scaled)) / 2048);
            };
        update_entry(corr_hist, pawn_key);
        update_entry(non_pawn_corr_w, np_w_key);
        update_entry(non_pawn_corr_b, np_b_key);
    }
};

// ============================================================================
// 5. 阶段式启发式走法排序器 (Staged MovePicker)
//    阶段顺序：TT_MOVE -> GOOD_CAPTURES -> KILLERS -> COUNTERMOVE / QUIETS -> BAD_CAPTURES
// ============================================================================

enum class PickerStage : uint8_t {
    TTMove = 0,
    GenerateMoves,
    GoodCaptures,
    Killer1,
    Killer2,
    CounterMove,
    ScoreQuiets,  // 新增：延迟安静走法打分阶段（仅在吃子和杀手走法均未截断时触发）
    QuietMoves,
    BadCaptures,
    Done
};

struct ScoredMove {
    Move    move  = Move::null();
    int32_t score = 0;
};

class MovePicker {
public:
    // 兼容原有的 7 参数构造（供静止搜索调用）
    MovePicker(
        const Position& pos,
        Move tt_move,
        int ply,
        const SearchHeuristics& heuristics,
        Piece prev_piece,
        Square prev_to,
        bool captures_only) noexcept
        : MovePicker(pos, tt_move, ply, heuristics, prev_piece, prev_to, Piece::None, Square::None, captures_only) {
    }

    // 支持 2-ply 延续历史的完整 9 参数构造
    MovePicker(
        const Position& pos,
        Move tt_move,
        int ply,
        const SearchHeuristics& heuristics,
        Piece prev_piece = Piece::None,
        Square prev_to = Square::None,
        Piece prev2_piece = Piece::None,
        Square prev2_to = Square::None,
        bool captures_only = false) noexcept
        : pos_(pos),
        tt_move_(tt_move),
        ply_(ply),
        heuristics_(heuristics),
        prev_piece_(prev_piece),
        prev_to_(prev_to),
        prev2_piece_(prev2_piece),
        prev2_to_(prev2_to),
        captures_only_(captures_only),
        stage_(PickerStage::TTMove)
    {
        if (ply_ >= 0 && ply_ < kMaxPly) {
            killer1_ = heuristics_.killers[static_cast<size_t>(ply_)][0];
            killer2_ = heuristics_.killers[static_cast<size_t>(ply_)][1];
        }
        counter_move_ = heuristics_.get_countermove(prev_piece_, prev_to_);
    }

    // 惰性按阶段挑选下一个走法；无须一次性全排序，若前序走法触发 Beta 截断则零后续排序开销
    [[nodiscard]] Move next_move(PickerStage* out_stage = nullptr) noexcept {
        while (stage_ != PickerStage::Done) {
            switch (stage_) {
            case PickerStage::TTMove: {
                stage_ = PickerStage::GenerateMoves;
                if (!tt_move_.is_none()) {
                    if (!captures_only_ || tt_move_.is_capture() || tt_move_.is_promotion()) {
                        // 极轻量基础校验：起点格必须有己方棋子，目标格不能是己方棋子
                        const Piece p = pos_.piece_on(tt_move_.from_sq());
                        if (p != Piece::None && piece_color(p) == pos_.side_to_move()) {
                            const Piece dest_p = pos_.piece_on(tt_move_.to_sq());
                            if (dest_p == Piece::None || piece_color(dest_p) != pos_.side_to_move()) {
                                if (out_stage) *out_stage = PickerStage::TTMove;
                                return tt_move_;
                            }
                        }
                    }
                }
                break;
            }

                case PickerStage::GenerateMoves: {
                    if (!generated_) {
                        generate_and_partition();
                    }
                    stage_ = PickerStage::GoodCaptures;
                    break;
                }

                case PickerStage::GoodCaptures: {
                    if (good_cap_idx_ < num_good_caps_) {
                        const Move m = pick_best(good_captures_, good_cap_idx_, num_good_caps_);
                        if (out_stage) *out_stage = PickerStage::GoodCaptures;
                        return m;
                    }
                    if (captures_only_) {
                        stage_ = PickerStage::Done;
                    } else {
                        stage_ = PickerStage::Killer1;
                    }
                    break;
                }

                case PickerStage::Killer1: {
                    stage_ = PickerStage::Killer2;
                    if (!killer1_.is_none() && killer1_ != tt_move_ && remove_quiet_move(killer1_)) {
                        if (out_stage) *out_stage = PickerStage::Killer1;
                        return killer1_;
                    }
                    break;
                }

                case PickerStage::Killer2: {
                    stage_ = PickerStage::CounterMove;
                    if (!killer2_.is_none() && killer2_ != tt_move_ && killer2_ != killer1_ &&
                        remove_quiet_move(killer2_))
                    {
                        if (out_stage) *out_stage = PickerStage::Killer2;
                        return killer2_;
                    }
                    break;
                }

                case PickerStage::CounterMove: {
                    // 试完反制走法后，流转到延迟打分阶段 ScoreQuiets
                    stage_ = PickerStage::ScoreQuiets;
                    if (!counter_move_.is_none() &&
                        counter_move_ != tt_move_ &&
                        counter_move_ != killer1_ &&
                        counter_move_ != killer2_ &&
                        remove_quiet_move(counter_move_))
                    {
                        if (out_stage) *out_stage = PickerStage::CounterMove;
                        return counter_move_;
                    }
                    break;
                }

                case PickerStage::ScoreQuiets: {
                    // 核心延迟机制：只有当 TT、好吃子、杀手1/2、反制全部未能截断时，才首次对安静走法打分！
                    score_quiet_moves();
                    stage_ = PickerStage::QuietMoves;
                    break;
                }

                case PickerStage::QuietMoves: {
                    if (quiet_idx_ < num_quiets_) {
                        const Move m = pick_best(quiets_, quiet_idx_, num_quiets_);
                        if (out_stage) *out_stage = PickerStage::QuietMoves;
                        return m;
                    }
                    stage_ = PickerStage::BadCaptures;
                    break;
                }

                case PickerStage::BadCaptures: {
                    if (bad_cap_idx_ < num_bad_caps_) {
                        const Move m = pick_best(bad_captures_, bad_cap_idx_, num_bad_caps_);
                        if (out_stage) *out_stage = PickerStage::BadCaptures;
                        return m;
                    }
                    stage_ = PickerStage::Done;
                    break;
                }

                case PickerStage::Done:
                    break;
            }
        }

        if (out_stage) *out_stage = PickerStage::Done;
        return Move::null();
    }

private:
    const Position&         pos_;
    Move                    tt_move_       = Move::null();
    int                     ply_           = 0;
    const SearchHeuristics& heuristics_;
    Piece                   prev_piece_    = Piece::None;
    Square                  prev_to_       = Square::None;
    Piece                   prev2_piece_   = Piece::None;
    Square                  prev2_to_      = Square::None;
    bool                    captures_only_ = false;
    PickerStage             stage_         = PickerStage::TTMove;
    bool                    generated_     = false;

    Move killer1_      = Move::null();
    Move killer2_      = Move::null();
    Move counter_move_ = Move::null();

    MoveList raw_list_{};
    std::array<ScoredMove, MoveList::kMaxMoves> good_captures_{};
    std::array<ScoredMove, MoveList::kMaxMoves> quiets_{};
    std::array<ScoredMove, MoveList::kMaxMoves> bad_captures_{};

    size_t num_good_caps_ = 0, good_cap_idx_ = 0;
    size_t num_quiets_    = 0, quiet_idx_    = 0;
    size_t num_bad_caps_  = 0, bad_cap_idx_  = 0;

    [[nodiscard]] bool contains_raw_move(Move target) const noexcept {
        for (Move m : raw_list_) {
            if (m == target) return true;
        }
        return false;
    }

    // 单趟選択排序：仅将 [cur_idx .. end_idx) 中最高分的走法交换到 cur_idx 并返回
    [[nodiscard]] static Move pick_best(
        std::array<ScoredMove, MoveList::kMaxMoves>& arr,
        size_t& cur_idx,
        size_t end_idx) noexcept
    {
        size_t best_i = cur_idx;
        int32_t best_s = arr[cur_idx].score;
        for (size_t i = cur_idx + 1; i < end_idx; ++i) {
            if (arr[i].score > best_s) {
                best_s = arr[i].score;
                best_i = i;
            }
        }
        if (best_i != cur_idx) {
            std::swap(arr[cur_idx], arr[best_i]);
        }
        return arr[cur_idx++].move;
    }

    // 若特殊安静走法（Killer / Countermove）存在于 quiets_ 池中，则将其提取出来避免重复搜索
    [[nodiscard]] bool remove_quiet_move(Move target) noexcept {
        for (size_t i = quiet_idx_; i < num_quiets_; ++i) {
            if (quiets_[i].move == target) {
                quiets_[i] = quiets_[--num_quiets_];
                return true;
            }
        }
        return false;
    }

    // 计算 MVV-LVA 得分：10 * VictimValue - AttackerValue + PromotionBonus
    [[nodiscard]] int32_t mvv_lva_score(Move m) const noexcept {
        const Piece attacker = pos_.piece_on(m.from_sq());
        int32_t victim_val = 0;
        if (m.is_en_passant()) {
            victim_val = piece_value(PieceType::Pawn);
        } else if (m.is_capture()) {
            victim_val = piece_value(pos_.piece_on(m.to_sq()));
        }
        int32_t promo_val = m.is_promotion() ? piece_value(m.promotion_type()) : 0;
        return victim_val * 10 + promo_val * 8 - piece_value(attacker);
    }

    // 延迟安静走法打分：仅在安静走法真正被需要时，才遍历打分
    void score_quiet_moves() noexcept {
        const Color us = pos_.side_to_move();
        for (size_t i = 0; i < num_quiets_; ++i) {
            const Move m = quiets_[i].move;
            int32_t q_score = heuristics_.get_history(us, m);
            if (prev_piece_ != Piece::None && prev_to_ != Square::None) {
                q_score += heuristics_.get_conthist(prev_piece_, prev_to_, pos_.piece_on(m.from_sq()), m.to_sq()) * 2;
            }
            if (prev2_piece_ != Piece::None && prev2_to_ != Square::None) {
                q_score += heuristics_.get_conthist2(prev2_piece_, prev2_to_, pos_.piece_on(m.from_sq()), m.to_sq());
            }
            const uint8_t to_f = file_of(m.to_sq());
            const uint8_t to_r = rank_of(m.to_sq());
            if (to_f >= 2U && to_f <= 5U && to_r >= 2U && to_r <= 5U) {
                q_score += 32;
            }
            quiets_[i].score = q_score;
        }
    }

    void generate_and_partition() noexcept {
        generated_ = true;
        generate_pseudo_legal_moves(pos_, raw_list_);
        const Color us = pos_.side_to_move();

        for (Move m : raw_list_) {
            if (m == tt_move_) continue;

            if (m.is_capture() || m.is_promotion()) {
                const Piece victim_p = m.is_en_passant()
                    ? make_piece(~us, PieceType::Pawn)
                    : pos_.piece_on(m.to_sq());
                const int32_t cap_hist = heuristics_.get_capture_history(
                    pos_.piece_on(m.from_sq()), m.to_sq(), piece_type(victim_p));

                const int32_t cap_score = mvv_lva_score(m) + cap_hist / 16;
                const bool is_obvious_good = (piece_value(pos_.piece_on(m.from_sq())) <= piece_value(victim_p));

                if (m.is_promotion() || is_obvious_good || see_ge(pos_, m, 0)) {
                    good_captures_[num_good_caps_++] = ScoredMove{ m, cap_score };
                }
                else if (!captures_only_) {
                    bad_captures_[num_bad_caps_++] = ScoredMove{ m, cap_score };
                }
            }
            else if (!captures_only_) {
                // 仅收集走法，分数赋 0，彻底省去 80% 节点的昂贵历史分计算！
                quiets_[num_quiets_++] = ScoredMove{ m, 0 };
            }
        }
    }
};

// ============================================================================
// 6. 搜索配置、时间控制器与搜索统计结果结构体
// ============================================================================

struct SearchLimits {
    int      max_depth = 64;
    uint64_t max_nodes = 0;     // 0 表示不限制节点数
    int64_t  soft_time_ms = 0;     // 迭代加深软时限（0 表示不限）
    int64_t  hard_time_ms = 0;     // 搜索内部硬时限（0 表示不限）
    bool     is_ponder = false; // 后台思考模式
    int      contempt = 0;     // 藐视和棋分数（正数代表激进拒绝和棋，负数代表求和）
    int      thread_id = 0;     // 线程编号（0 为主线程，1..N-1 为辅助线程）
};

struct SearchStats {
    uint64_t          nodes      = 0;
    uint64_t          qnodes     = 0;
    uint64_t          tt_hits    = 0;
    int               depth      = 0;
    int               seldepth   = 0;
    int               best_score = 0;
    Move              best_move  = Move::null();
    int64_t           time_ms    = 0;
    uint64_t          nps        = 0;
    std::vector<Move> pv_line{};

    // 剪枝触发统计（用于分析与验证）
    uint64_t rfp_prunes       = 0;
    uint64_t nmp_prunes       = 0;
    uint64_t lmp_prunes       = 0;
    uint64_t lmr_researches   = 0;
    uint64_t singular_extends = 0;
};

// LMR 对数减深查找表（编译期或首次构造时初始化）
struct LmrTable {
    std::array<std::array<int, 64>, 64> reductions{};

    LmrTable() noexcept {
        for (int d = 1; d < 64; ++d) {
            for (int m = 1; m < 64; ++m) {
                reductions[static_cast<size_t>(d)][static_cast<size_t>(m)] =
                    static_cast<int>(0.85 + std::log(static_cast<double>(d)) * std::log(static_cast<double>(m)) / 1.95);
            }
        }
    }

    [[nodiscard]] int get(int depth, int move_count) const noexcept {
        const size_t d = static_cast<size_t>(std::clamp(depth, 0, 63));
        const size_t m = static_cast<size_t>(std::clamp(move_count, 0, 63));
        return reductions[d][m];
    }
};

inline const LmrTable kLmrTable{};

// ============================================================================
// 7. 零侵入 O(1) 空步执行器 (NullMoveGuard)
//    自动兼容 Stage 1 Position 类（利用非易位敏感格的原地走子翻转走子方、清除过路兵、压入历史哈希）
// ============================================================================

class NullMoveGuard {
public:
    NullMoveGuard(Position& pos, StateInfo& st) noexcept : pos_(pos), st_(st) {
        pos_.make_null_move(st_);
        active_ = true;
    }

    [[nodiscard]] bool active() const noexcept { return active_; }

    void undo() noexcept {
        if (active_) {
            pos_.undo_null_move(st_);
            active_ = false;
        }
    }

private:
    Position&  pos_;
    StateInfo& st_;
    bool       active_ = false;
};

// ============================================================================
// 8. Searcher 核心搜索引擎类 (PVS + Quiescence + RFP + NMP + LMR + LMP + SE)
// ============================================================================

class Searcher {
public:
    explicit Searcher(size_t tt_size_mb = 16)
        : tt_(std::make_shared<TranspositionTable>(tt_size_mb)),
          evaluator_(std::make_unique<nnue::Evaluator>()),
          ctx_(std::make_unique<SearchContext>())
    {
        ctx_->heuristics.clear();
    }

    Searcher(size_t tt_size_mb, std::shared_ptr<nnue::Evaluator> shared_eval)
        : tt_(std::make_shared<TranspositionTable>(tt_size_mb)),
          evaluator_(std::move(shared_eval)),
          ctx_(std::make_unique<SearchContext>())
    {
        if (!evaluator_) {
            evaluator_ = std::make_unique<nnue::Evaluator>();
        }
        ctx_->heuristics.clear();
    }

    Searcher(std::shared_ptr<TranspositionTable> shared_tt, std::shared_ptr<nnue::Evaluator> shared_eval)
        : tt_(std::move(shared_tt)),
          evaluator_(std::move(shared_eval)),
          ctx_(std::make_unique<SearchContext>())
    {
        if (!tt_) {
            tt_ = std::make_shared<TranspositionTable>(16);
        }
        if (!evaluator_) {
            evaluator_ = std::make_unique<nnue::Evaluator>();
        }
        ctx_->heuristics.clear();
    }

    void resize_tt_mb(size_t mb) { tt_->resize_mb(mb); }
    void clear_tt() noexcept { tt_->clear(); }
    void clear_heuristics() noexcept { ctx_->heuristics.clear(); }

    void notify_ponderhit(int64_t soft_ms, int64_t hard_ms) noexcept {
        start_time_ = std::chrono::steady_clock::now();
        limits_.soft_time_ms = soft_ms;
        limits_.hard_time_ms = hard_ms;
        is_ponder_.store(false, std::memory_order_relaxed);
    }

    void request_stop() noexcept { stop_flag_.store(true, std::memory_order_relaxed); }
    [[nodiscard]] bool is_stopped() const noexcept { return stop_flag_.load(std::memory_order_relaxed); }

    [[nodiscard]] const TranspositionTable& tt() const noexcept { return *tt_; }
    [[nodiscard]] TranspositionTable& tt() noexcept { return *tt_; }
    [[nodiscard]] std::shared_ptr<TranspositionTable> shared_tt() const noexcept { return tt_; }
    [[nodiscard]] const nnue::Evaluator& evaluator() const noexcept { return *evaluator_; }
    [[nodiscard]] nnue::Evaluator& evaluator() noexcept { return *evaluator_; }
    [[nodiscard]] const SearchStats& stats() const noexcept { return stats_; }

    // ------------------------------------------------------------------------
    // 迭代加深 (Iterative Deepening) + 期望窗口 (Aspiration Windows) 主入口
    // ------------------------------------------------------------------------
    SearchStats search_position(
        Position& pos,
        const SearchLimits& limits,
        const std::function<void(const SearchStats&)>& on_depth_finish = nullptr,
        bool is_main_thread = true) {
        stop_flag_.store(false, std::memory_order_relaxed);
        is_ponder_.store(limits.is_ponder, std::memory_order_relaxed);
        limits_     = limits;
        start_time_ = std::chrono::steady_clock::now();
        root_color_ = pos.side_to_move(); // <--- 新增这行：记录根节点执棋方！

        stats_ = SearchStats{};
        nmp_min_ply_ = 0;
        if (is_main_thread) {
            tt_->new_search();
            ctx_->heuristics.age_history();
        }

        // 初始化根节点 NNUE 累加器栈
        evaluator_->refresh_accumulator(pos, ctx_->acc_stack[0], &ctx_->finny_table);
        for (auto& node : ctx_->node_stack) {
            node = PlyNodeState{};
        }

        int prev_score     = 0;
        Move best_root_mv  = Move::null();
        std::vector<Move> best_pv{};
        int move_stability = 0;

        const int max_d = std::clamp(limits_.max_depth, 1, kMaxPly - 2);

        for (int depth = 1; depth <= max_d; ++depth) {
            if (!is_main_thread && depth > 1 && ((depth + limits_.thread_id) % 3 == 0)) {
                continue;
            }
            ctx_->pv_length.fill(0);

            // 放宽初始窗口至 65（约 31 cp），兼顾快速剪枝与首查命中率
            int delta = 65;
            int alpha = -kInfiniteScore;
            int beta = kInfiniteScore;

            // 深度 >= 4 时开启期望窗口
            if (depth >= 4 && !is_mate_score(prev_score)) {
                alpha = std::max(-kInfiniteScore, prev_score - delta);
                beta = std::min(kInfiniteScore, prev_score + delta);
            }

            int score = 0;
            int fail_count = 0;

            while (true) {
                score = pvs<true>(pos, depth, 0, alpha, beta, false);
                if (is_stopped()) break;

                // 当前搜索窗口已经是全窗口时，搜索结果为真实极限值，直接退出
                if (alpha <= -kInfiniteScore && beta >= kInfiniteScore) {
                    break;
                }

                if (score <= alpha) {
                    // Fail-Low 智能重居中：直接以实际跌落的 score 为基准锚定并拓宽，杜绝连续多次无效试探
                    beta = (alpha + beta) / 2; // 适度下压 beta 窗口上限，收紧剪枝带
                    alpha = std::max(-kInfiniteScore, score - delta);
                    delta += delta / 2 + delta / 4;
                    ++fail_count;
                }
                else if (score >= beta) {
                    // Fail-High 智能重居中：直接以实际突破的 score 为基准锚定并拓宽
                    alpha = (alpha + beta) / 2; // 适度提升 alpha 窗口下限，加速后续截断
                    beta = std::min(kInfiniteScore, score + delta);
                    delta += delta / 2 + delta / 4;
                    ++fail_count;
                }
                else {
                    // 分数刚好落在窗口内部，完美一遍过！
                    break;
                }

                // 连续失误 3 次或步长超出 350 时，立即全开窗口一步到位，杜绝 6~8 次反复试探
                if (fail_count >= 3 || delta > 350) {
                    alpha = -kInfiniteScore;
                    beta = kInfiniteScore;
                    score = pvs<true>(pos, depth, 0, alpha, beta, false);
                    break;
                }
            }

            if (is_stopped()) {
                break;
            }

            const int last_score = prev_score;
            prev_score = score;
            if (ctx_->pv_length[0] > 0 && !ctx_->pv_table[0][0].is_none()) {
                const Move new_best = ctx_->pv_table[0][0];
                if (new_best == best_root_mv) {
                    ++move_stability;
                } else {
                    move_stability = 0;
                }
                best_root_mv = new_best;
                best_pv.assign(
                    ctx_->pv_table[0].begin(),
                    ctx_->pv_table[0].begin() + static_cast<ptrdiff_t>(ctx_->pv_length[0]));
            }

            const auto now = std::chrono::steady_clock::now();
            const int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();

            stats_.depth      = depth;
            stats_.best_score = prev_score;
            stats_.best_move  = best_root_mv;
            stats_.pv_line    = best_pv;
            if (on_depth_finish) {
                on_depth_finish(stats_);
            }

            // ▼▼▼ 新增：必胜杀棋提前跳出机制 ▼▼▼
            // 若当前已证明了强制杀棋（正杀棋分），且搜索深度已达到杀棋步数：
            // 杀棋序列已 100% 算穿到底，世界上没有任何一步棋能比这更优，立即退出出步！
            if (prev_score >= kMateThreshold) {
                const int mate_plies = kMateScore - prev_score;
                if (depth >= mate_plies) {
                    break; // 毫秒级闪电出招，绝不再进下一层无谓耗时！
                }
            }
            // ▲▲▲ 新增结束 ▲▲▲

            stats_.time_ms    = elapsed_ms;
            stats_.nps        = (elapsed_ms > 0)
                ? (stats_.nodes * 1000ULL) / static_cast<uint64_t>(elapsed_ms)
                : stats_.nodes;

            // 自适应动态步时控制 2.1 (高低时限平衡 + 解耦稳定性与可行性早停):
            if (!is_ponder_.load(std::memory_order_relaxed) && limits_.soft_time_ms > 0) {
                int64_t target_soft = limits_.soft_time_ms;
                const bool low_time_budget = (limits_.soft_time_ms < 1500);

                // 1. 基于 move_stability 的动态伸缩因子
                if (depth >= 5 && !is_mate_score(prev_score)) {
                    if (move_stability == 0) {
                        // 战术危机（刚刚突发换招）：短时限封顶 1.5x 防透支，长时限允许 2.2x 深算
                        const int64_t crisis_pct = low_time_budget ? 150 : 220;
                        target_soft = (target_soft * crisis_pct) / 100;
                    }
                    else if (move_stability == 1) {
                        // 换招再确认阶段：短时限 1.25x，常规 1.4x
                        const int64_t recheck_pct = low_time_budget ? 125 : 140;
                        target_soft = (target_soft * recheck_pct) / 100;
                    }
                    else if (move_stability >= 4) {
                        // 走法连续 4 层稳固：从 0.55x 放宽至 0.75x，避免与可行性早停叠乘过度省时
                        target_soft = (target_soft * 75) / 100;
                    }
                    else if (move_stability >= 2) {
                        // 走法正常平稳：微幅节约 10%
                        target_soft = (target_soft * 90) / 100;
                    }

                    // 局势突发暴跌（失分超过 35 cp）：短时限封顶 1.4x，常规 1.8x 防御性长考
                    if (depth >= 3 && last_score - score >= 35) {
                        const int64_t drop_pct = low_time_budget ? 140 : 180;
                        target_soft = std::max(target_soft, (limits_.soft_time_ms * drop_pct) / 100);
                    }
                }

                // 硬时限安全红线保护：无论如何放大，绝不超出 hard_time - 25ms，防超时判负
                if (limits_.hard_time_ms > 0) {
                    const int64_t safe_ceiling = std::max<int64_t>(10, limits_.hard_time_ms - 25);
                    target_soft = std::min(target_soft, safe_ceiling);
                }

                // 2. 预测下一层完成度（Next-Depth Feasibility Check）：
                // 将早停阈值从 55% 调整至 68%，与稳定走法 0.75x 配合后实际约 0.51x，兼顾省时与深度
                if (elapsed_ms >= target_soft || (depth >= 6 && elapsed_ms * 100 >= target_soft * 68)) {
                    break;
                }
            }
        }

        // 兜底：若 depth 1 因极短时限未产出 best_move，则挑选第一个合法走法
        if (stats_.best_move.is_none()) {
            MoveList root_moves;
            generate_legal_moves(pos, root_moves);
            if (!root_moves.empty()) {
                stats_.best_move = root_moves[0];
                stats_.pv_line   = {root_moves[0]};
            }
        }

        const auto end_t = std::chrono::steady_clock::now();
        stats_.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_time_).count();
        stats_.nps     = (stats_.time_ms > 0)
            ? (stats_.nodes * 1000ULL) / static_cast<uint64_t>(stats_.time_ms)
            : stats_.nodes;
        return stats_;
    }

private:
    struct PlyNodeState {
        int    static_eval   = kValueNone;
        Move   current_move  = Move::null();
        Piece  moved_piece   = Piece::None;
        Move   excluded_move = Move::null();
    };

    struct SearchContext {
        SearchHeuristics                                heuristics{};
        std::array<nnue::Accumulator, kMaxPly + 2>      acc_stack{};
        nnue::FinnyTable                                finny_table{};
        std::array<PlyNodeState,      kMaxPly + 2>      node_stack{};
        std::array<std::array<Move, kMaxPly>, kMaxPly>  pv_table{};
        std::array<size_t, kMaxPly>                     pv_length{};
    };

    std::shared_ptr<TranspositionTable> tt_;
    std::shared_ptr<nnue::Evaluator>    evaluator_;
    std::unique_ptr<SearchContext>      ctx_;
    SearchLimits                        limits_{};
    SearchStats                         stats_{};
    std::atomic<bool>                   stop_flag_{false};
    std::atomic<bool>                   is_ponder_{false};
    std::chrono::steady_clock::time_point start_time_{};
    Color root_color_ = Color::White;
    int   nmp_min_ply_ = 0; // 高深度 NMP 验证搜索期间禁用空步剪枝的层数下限

    // 动态计算当前节点和棋分（解决 Negamax 符号奇偶层颠倒问题）
    [[nodiscard]] int draw_score(const Position& pos, int ply) const noexcept {
        (void)ply;
        if (limits_.contempt == 0) return kDrawScore;
        // 若当前走子方是我方（根节点方），和棋对我方不利，返回 -contempt
        // 若当前走子方是对手，和棋对对手不利，从对手视角看相当于我方得分，返回 +contempt
        const bool is_root_side = (pos.side_to_move() == root_color_);
        return is_root_side ? -limits_.contempt : limits_.contempt;
    }

    // 每隔 2048 节点检查一次硬时间上限与最大节点上限
    void check_time_and_limits() noexcept {
        if (limits_.max_nodes > 0 && stats_.nodes >= limits_.max_nodes) {
            stop_flag_.store(true, std::memory_order_relaxed);
            return;
        }
        const uint64_t mask = (limits_.hard_time_ms > 0 && limits_.hard_time_ms < 1500) ? 255ULL : 2047ULL;
        if ((stats_.nodes & mask) == 0ULL && limits_.hard_time_ms > 0 && !is_ponder_.load(std::memory_order_relaxed)) {
            const auto now = std::chrono::steady_clock::now();
            const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
            if (elapsed >= limits_.hard_time_ms) {
                stop_flag_.store(true, std::memory_order_relaxed);
            }
        }
    }

    void update_pv(int ply, Move m) noexcept {
        const auto p = static_cast<size_t>(ply);
        if (p + 1 >= kMaxPly) return;
        ctx_->pv_table[p][p] = m;
        const size_t child_end = std::min<size_t>(kMaxPly, ctx_->pv_length[p + 1]);
        for (size_t next = p + 1; next < child_end; ++next) {
            ctx_->pv_table[p][next] = ctx_->pv_table[p + 1][next];
        }
        ctx_->pv_length[p] = std::max(p + 1, child_end);
    }

    // ------------------------------------------------------------------------
    // 静态搜索 (Quiescence Search): Stand-pat 剪枝 + 吃子/升变搜索消除地平线效应
    // ------------------------------------------------------------------------
    [[nodiscard]] int quiescence(Position& pos, int ply, int alpha, int beta, int qs_depth = 0) {
        if (ply > stats_.seldepth) {
            stats_.seldepth = ply;
        }

        check_time_and_limits();
        if (is_stopped()) return 0;

        // 50 回合与三次重复判和
        if (ply > 0 && (pos.halfmove_clock() >= 100 || pos.is_repetition_draw(ply))) {
            return draw_score(pos, ply);
        }

        if (ply >= kMaxPly - 1) {
            return evaluator_->evaluate(pos, ctx_->acc_stack[static_cast<size_t>(ply)]);
        }

        const bool is_in_check = in_check(pos);

        // 1. 置换表探测
        TTEntry tte{};
        Move tt_move = Move::null();
        if (tt_->probe(pos.zobrist_key(), tte)) {
            ++stats_.tt_hits;
            tt_move = tte.move();
            const int tt_score = score_from_tt(tte.score, ply);
            if (tte.bound() == TTBound::Exact ||
                (tte.bound() == TTBound::LowerBound && tt_score >= beta) ||
                (tte.bound() == TTBound::UpperBound && tt_score <= alpha))
            {
                return tt_score;
            }
        }

        // 2. Stand-pat 静态评估剪枝（仅当未被将军时可用）
        int stand_pat = -kInfiniteScore;
        int best_score = -kInfiniteScore;
        if (!is_in_check) {
            stand_pat = (tte.bound() != TTBound::None && tte.static_eval != kValueNone)
                ? static_cast<int>(tte.static_eval)
                : evaluator_->evaluate(pos, ctx_->acc_stack[static_cast<size_t>(ply)]);
            best_score = stand_pat;

            if (stand_pat >= beta) {
                // 缓存静态截断分至置换表：转置到相同局面时 5 纳秒直接命中，免去整套 NNUE 神经网络推理！
                tt_->store(pos.zobrist_key(), Move::null(), stand_pat, stand_pat, 0, TTBound::LowerBound, ply);
                return stand_pat;
            }
            if (stand_pat > alpha) {
                alpha = stand_pat;
            }
        }

        // 3. 挑选吃子与升变走法（若被将军则生成全部逃将走法）
        const Piece  prev_piece = (ply > 0) ? ctx_->node_stack[static_cast<size_t>(ply - 1)].moved_piece : Piece::None;
        const Square prev_to    = (ply > 0 && !ctx_->node_stack[static_cast<size_t>(ply - 1)].current_move.is_none())
            ? ctx_->node_stack[static_cast<size_t>(ply - 1)].current_move.to_sq()
            : Square::None;

        const bool allow_evasions = is_in_check;
        MovePicker picker(pos, tt_move, ply, ctx_->heuristics, prev_piece, prev_to, !allow_evasions);

        Move best_move = Move::null();
        int legal_moves = 0;

        while (true) {
            const Move m = picker.next_move();
            if (m.is_none()) break;

            // Delta Pruning: 若非将军、非升变，且吃掉目标子加上 360 内部单位（约 1 兵）裕量仍不足以提升 alpha，则跳过
            if (!is_in_check && !m.is_promotion()) {
                const int gain = m.is_en_passant()
                    ? piece_value(PieceType::Pawn)
                    : piece_value(pos.piece_on(m.to_sq()));
                if (stand_pat + gain + 360 < alpha) {
                    continue;
                }

                // QSearch SEE 剪枝：静止搜索坚决不搜亏损吃子 (SEE < 0)，直接剪枝且跳过合法性检测！
                if (!see_ge(pos, m, 0)) {
                    continue;
                }
            }

            if (!is_legal(pos, m)) continue;
            ++legal_moves;

            // TT 硬件缓存行预取：提前将静止搜索子局面的 TT Cluster 拉入 L1 Cache
            tt_->prefetch(approximate_key_after(pos, m));

            ++stats_.nodes;
            ++stats_.qnodes;

            // 拷贝当前层 Accumulator 至下一层并增量更新（悔棋时直接丢弃子层 Accumulator，零回滚开销）
            const Color us    = pos.side_to_move();
            const Piece moved = pos.piece_on(m.from_sq());
            StateInfo st{};
            pos.make_move(m, st);

            const nnue::DirtyFeatures df = nnue::extract_dirty_features(us, moved, m, st);
            evaluator_->update_on_make_move(pos, df, ctx_->acc_stack[static_cast<size_t>(ply)], ctx_->acc_stack[static_cast<size_t>(ply + 1)], &ctx_->finny_table);

            ctx_->node_stack[static_cast<size_t>(ply)].current_move = m;
            ctx_->node_stack[static_cast<size_t>(ply)].moved_piece  = moved;

            const int score = -quiescence(pos, ply + 1, -beta, -alpha, qs_depth + 1);
            pos.undo_move(m, st);

            if (is_stopped()) return 0;

            if (score > best_score) {
                best_score = score;
                best_move  = m;
                if (score > alpha) {
                    alpha = score;
                    if (score >= beta) {
                        tt_->store(pos.zobrist_key(), best_move, best_score, stand_pat, 0, TTBound::LowerBound, ply);
                        return best_score;
                    }
                }
            }
        }

        // 若被将军且无任何合法逃将走法，则判定为被将死
        if (is_in_check && legal_moves == 0) {
            return -kMateScore + ply;
        }

        const TTBound q_bound = (best_score >= beta) ? TTBound::LowerBound : TTBound::UpperBound;
        tt_->store(pos.zobrist_key(), best_move, best_score, stand_pat, 0, q_bound, ply);
        return best_score;
    }

    // ------------------------------------------------------------------------
    // Alpha-Beta / PVS 主搜索核心 (含 RFP, NMP, Singular Extensions, LMP, LMR)
    // ------------------------------------------------------------------------
    template <bool IsPV>
    [[nodiscard]] int pvs(Position& pos, int depth, int ply, int alpha, int beta, bool cut_node) {
        const auto ply_idx = static_cast<size_t>(ply);
        ctx_->pv_length[ply_idx] = ply_idx;

        if (ply > stats_.seldepth) {
            stats_.seldepth = ply;
        }

        const bool is_root = (ply == 0);
        const Move excluded_move = ctx_->node_stack[ply_idx].excluded_move;

        // 1. 叶节点进入 Quiescence Search
        if (depth <= 0) {
            return quiescence(pos, ply, alpha, beta);
        }

        check_time_and_limits();
        if (is_stopped()) return 0;

        if (!is_root) {
            // 50 回合与重复局面判和（精确区分树内循环与真实历史）
            if (pos.halfmove_clock() >= 100 || pos.is_repetition_draw(ply)) {
                return draw_score(pos, ply);
            }

            // Mate Distance Pruning (杀棋步数距离剪枝)
            alpha = std::max(alpha, -kMateScore + ply);
            beta  = std::min(beta,   kMateScore - ply - 1);
            if (alpha >= beta) {
                return alpha;
            }
        }

        if (ply >= kMaxPly - 1) {
            return evaluator_->evaluate(pos, ctx_->acc_stack[ply_idx]);
        }

        const bool is_in_check = in_check(pos);

        // 2. 置换表探测 (TT Probe)
        TTEntry tte{};
        bool tt_hit  = false;
        Move tt_move = Move::null();
        int  tt_score = kValueNone;

        if (excluded_move.is_none() && tt_->probe(pos.zobrist_key(), tte)) {
            tt_hit   = true;
            ++stats_.tt_hits;
            tt_move  = tte.move();
            tt_score = score_from_tt(tte.score, ply);

            // 非 PV 节点且 TT 深度充足时直接截断返回
            if (!IsPV && tte.depth >= depth) {
                if (tte.bound() == TTBound::Exact ||
                    (tte.bound() == TTBound::LowerBound && tt_score >= beta) ||
                    (tte.bound() == TTBound::UpperBound && tt_score <= alpha))
                {
                    return tt_score;
                }
            }
        }

        // 3. 静态评估计算与 Improving 趋势判定（raw_eval 存原始评估，static_eval 存校准后评估）
        int raw_eval = kValueNone;
        int static_eval = kValueNone;
        if (!is_in_check) {
            if (tt_hit && tte.static_eval != kValueNone) {
                raw_eval = tte.static_eval;
            }
            else {
                raw_eval = evaluator_->evaluate(pos, ctx_->acc_stack[ply_idx]);
            }
            // CorrHist 多源动态校准：结合兵型与双方非兵子力偏差修正 static_eval（绝不污染 raw_eval）
            const int corr = ctx_->heuristics.get_correction(
                pos.side_to_move(),
                compute_pawn_key(pos),
                compute_non_pawn_key(pos, Color::White),
                compute_non_pawn_key(pos, Color::Black));
            static_eval = std::clamp(raw_eval + corr, -kMateThreshold + 1, kMateThreshold - 1);
            ctx_->node_stack[ply_idx].static_eval = static_eval;
        }
        else {
            ctx_->node_stack[ply_idx].static_eval = kValueNone;
        }

        const bool improving = (!is_in_check && ply >= 2 &&
            ctx_->node_stack[ply_idx - 2].static_eval != kValueNone &&
            static_eval > ctx_->node_stack[ply_idx - 2].static_eval);

        // 提前提取上一层与上上层历史走子（用于延续历史、反制走法与 ProbCut 排序）
        const Piece  prev_piece = (ply > 0) ? ctx_->node_stack[ply_idx - 1].moved_piece : Piece::None;
        const Square prev_to = (ply > 0 && !ctx_->node_stack[ply_idx - 1].current_move.is_none())
            ? ctx_->node_stack[ply_idx - 1].current_move.to_sq()
            : Square::None;
        const Piece  prev2_piece = (ply >= 2) ? ctx_->node_stack[ply_idx - 2].moved_piece : Piece::None;
        const Square prev2_to = (ply >= 2 && !ctx_->node_stack[ply_idx - 2].current_move.is_none())
            ? ctx_->node_stack[ply_idx - 2].current_move.to_sq()
            : Square::None;

        // 4. 非 PV、非将军、非奇异搜索下的前向剪枝 (ProbCut, Razoring, RFP & NMP)
        if (!IsPV && !is_in_check && excluded_move.is_none()) {

            // (A-1) ProbCut (概率快速前向剪枝): 若以更高 beta 门槛做浅层搜索依然截断，直接判定胜势截断！
            if (false && depth >= 5 && std::abs(beta) < kMateThreshold) {
                const int prob_beta = std::min(kInfiniteScore - 1, beta + 180);
                const int prob_depth = depth - 4;

                // 严谨零副作用短路：只有当置换表明确给出 UpperBound（得分上限）或 Exact 且分值确认低于 prob_beta 时才跳过
                const bool tt_proves_failed = (tt_hit && tte.depth >= prob_depth && tt_score < prob_beta &&
                    (tte.bound() == TTBound::UpperBound || tte.bound() == TTBound::Exact));

                if (!tt_proves_failed) {
                    MovePicker prob_picker(pos, tt_move, ply, ctx_->heuristics, prev_piece, prev_to, prev2_piece, prev2_to, true);
                    while (true) {
                        const Move pm = prob_picker.next_move();
                        if (pm.is_none()) break;
                        if (!see_ge(pos, pm, prob_beta - static_eval)) continue;
                        if (!is_legal(pos, pm)) continue;

                        const Color prob_us = pos.side_to_move();
                        const Piece prob_moved = pos.piece_on(pm.from_sq());
                        StateInfo prob_st{};
                        pos.make_move(pm, prob_st);
                        const nnue::DirtyFeatures df = nnue::extract_dirty_features(prob_us, prob_moved, pm, prob_st);
                        evaluator_->update_on_make_move(pos, df, ctx_->acc_stack[ply_idx], ctx_->acc_stack[ply_idx + 1], &ctx_->finny_table);

                        int prob_score = -quiescence(pos, ply + 1, -prob_beta, -prob_beta + 1);
                        if (prob_score >= prob_beta && prob_depth > 0) {
                            prob_score = -pvs<false>(pos, prob_depth, ply + 1, -prob_beta, -prob_beta + 1, !cut_node);
                        }
                        pos.undo_move(pm, prob_st);

                        if (is_stopped()) return 0;
                        if (prob_score >= prob_beta) {
                            tt_->store(pos.zobrist_key(), pm, prob_score, static_eval, depth - 3, TTBound::LowerBound, ply);
                            return prob_score;
                        }
                    } // 闭合 while (true)
                }     // 闭合 if (!tt_proves_failed)
            }         // 闭合 if (depth >= 5)

            // (A0) Razoring (浅层剃刀剪枝): 对极度劣势的浅层节点(depth <= 3)，用静止搜索探底快速剪枝
            if (depth <= 3 &&
                static_eval != kValueNone &&
                std::abs(alpha) < kMateThreshold &&
                static_eval + 200 + 180 * depth <= alpha)
            {
                const int q_score = quiescence(pos, ply, alpha, beta);
                if (q_score <= alpha) {
                    return q_score; // 连吃子反抗都救不回局势，直接把整棵子树一刀切掉！
                }
            }

            // (A) RFP (Reverse Futility Pruning / 静态空步剪枝)
            const int rfp_margin = (improving ? 105 : 150) * depth;
            if (depth <= 8 &&
                static_eval - rfp_margin >= beta &&
                std::abs(beta) < kMateThreshold &&
                std::abs(static_eval) < kMateThreshold)
            {
                ++stats_.rfp_prunes;
                return beta;
            }

            // (B) NMP (Null Move Pruning / 空步剪枝)
            // 自适应减深 R = 3 + depth / 6 + min(3, (static_eval - beta) / 200)
            // 带残局非兵子力过滤 (has_non_pawn_material) 与将死威胁检测
            const bool prev_was_null = (ply > 0 && ctx_->node_stack[ply_idx - 1].current_move.is_none());
            if (depth >= 3 &&
                ply >= nmp_min_ply_ &&
                !prev_was_null &&
                static_eval >= beta &&
                std::abs(beta) < kMateThreshold &&
                has_non_pawn_material(pos, pos.side_to_move()))
            {
                const int r = 3 + depth / 6 + std::min(3, (static_eval - beta) / 200);
                const int null_depth = std::max(1, depth - r);

                StateInfo null_st{};
                NullMoveGuard null_guard(pos, null_st);
                if (null_guard.active()) {
                    // 零拷贝优化：空步盘面棋子完全不动，子层直接复用当前层累加器，消灭 2KB 冗余 memcpy
                    ctx_->acc_stack[ply_idx + 1].white = ctx_->acc_stack[ply_idx].white;
                    ctx_->acc_stack[ply_idx + 1].black = ctx_->acc_stack[ply_idx].black;
                    ctx_->acc_stack[ply_idx + 1].white_king_sq = ctx_->acc_stack[ply_idx].white_king_sq;
                    ctx_->acc_stack[ply_idx + 1].black_king_sq = ctx_->acc_stack[ply_idx].black_king_sq;
                    ctx_->acc_stack[ply_idx + 1].computed_white = ctx_->acc_stack[ply_idx].computed_white;
                    ctx_->acc_stack[ply_idx + 1].computed_black = ctx_->acc_stack[ply_idx].computed_black;

                    ctx_->node_stack[ply_idx].current_move = Move::null();
                    ctx_->node_stack[ply_idx].moved_piece = Piece::None;

                    const int null_score = -pvs<false>(pos, null_depth, ply + 1, -beta, -beta + 1, !cut_node);
                    null_guard.undo();

                    if (is_stopped()) return 0;

                    if (null_score >= beta) {
                        if (depth < 12 || nmp_min_ply_ > 0) {
                            ++stats_.nmp_prunes;
                            return beta; // 严格返回 beta，杜绝虚高评估污染父节点窗口
                        }
                        // 高深度验证搜索：禁用当前子树 NMP，确认不走棋依然 >= beta 再截断
                        nmp_min_ply_ = ply + (3 * (depth - r)) / 4;
                        const int verify_score = pvs<false>(pos, depth - r, ply, beta - 1, beta, false);
                        nmp_min_ply_ = 0;

                        if (is_stopped()) return 0;
                        if (verify_score >= beta) {
                            ++stats_.nmp_prunes;
                            return beta;
                        }
                    }
                }
            }
        }

        // 5. 阶段式走法枚举与 PVS 主循环
        // (B2) IIR (Internal Iterative Reductions): 缺少置换表引导时浅搜 1 层建立粗略排序
        if (depth >= 4 && tt_move.is_none() && (IsPV || cut_node)) {
            --depth;
        }

        MovePicker picker(pos, tt_move, ply, ctx_->heuristics, prev_piece, prev_to, prev2_piece, prev2_to, false);

        int  best_score = -kInfiniteScore;
        Move best_move = Move::null();
        int  legal_moves = 0;
        int  move_count = 0;
        int  quiet_count = 0;
        const int orig_alpha = alpha;

        std::array<Move, 64> searched_quiets{};
        size_t num_searched_quiets = 0;

        while (true) {
            PickerStage stage = PickerStage::Done;
            const Move m = picker.next_move(&stage);
            if (m.is_none()) break;
            if (m == excluded_move) continue;

            const bool is_quiet = (!m.is_capture() && !m.is_promotion());
            ++move_count;

            // 1. 剪枝安全屏障：首个候选走法（move_count == 1）以及将军局面严禁任何前向剪枝！
            if (!IsPV && !is_in_check && move_count > 1 && best_score > -kMateThreshold) {

                // (C1) FP (Futility Pruning): 动态估值差距剪枝
                if (is_quiet && depth <= 8) {
                    const int fp_margin = (improving ? 200 : 140) * depth;
                    if (static_eval != kValueNone && static_eval + fp_margin <= alpha) {
                        continue;
                    }
                }

                // (C1.5) Quiet SEE Pruning: 过滤白送子的劣质安静走法
                if (is_quiet && depth <= 8) {
                    if (!see_ge(pos, m, -35 * depth * depth)) {
                        continue;
                    }
                }

                // (C1.6) Bad Capture SEE Pruning: 过滤白送大子的严重亏子吃子
                if (stage == PickerStage::BadCaptures && depth <= 6) {
                    if (!see_ge(pos, m, -100 * depth)) {
                        continue;
                    }
                }

                // (C2) LMP (Late Move Pruning / 晚步剪枝): 仅统计实际搜索的合法安静走法数
                if (is_quiet && depth <= 8) {
                    const int lmp_limit = (4 + depth * depth) / (2 - (improving ? 1 : 0));
                    if (quiet_count >= lmp_limit) {
                        ++stats_.lmp_prunes;
                        continue;
                    }
                }

                // (C3) History Pruning (历史表综合负反馈剪枝)
                if (is_quiet && depth <= 4) {
                    int32_t total_h = ctx_->heuristics.get_history(pos.side_to_move(), m);
                    if (prev_piece != Piece::None && prev_to != Square::None) {
                        total_h += ctx_->heuristics.get_conthist(prev_piece, prev_to, pos.piece_on(m.from_sq()), m.to_sq()) * 2;
                    }
                    // 仅当包含延续历史在内的综合历史表现极度恶劣时才剪枝，杜绝误杀防守反击妙手
                    if (total_h < -2560 * depth) {
                        continue;
                    }
                }
            }

            // 2. 只有通过了上面所有剪枝、真正需要深入搜索的走法，才进行合法性检查！
            // 大幅规避了对被剪枝安静走法的昂贵射线计算
            if (!is_legal(pos, m)) continue;
            ++legal_moves;
            if (is_quiet) ++quiet_count;
            ++stats_.nodes;

            // 3. 置换表硬件预取：提前把下层局面的 TT Cluster 拉入 L1 缓存
            tt_->prefetch(approximate_key_after(pos, m));

            // (D) Singular Extensions (奇异延伸测试框架) + 将军延伸
            int extension = 0;
            if (!is_root &&
                depth >= 8 &&
                m == tt_move &&
                excluded_move.is_none() &&
                tt_hit &&
                tte.depth >= depth - 3 &&
                (tte.bound() == TTBound::LowerBound || tte.bound() == TTBound::Exact) &&
                std::abs(tt_score) < kMateThreshold)
            {
                const int singular_beta = tt_score - 2 * depth;
                const int singular_depth = (depth - 1) / 2;

                ctx_->node_stack[ply_idx].excluded_move = m;
                const int se_score = pvs<false>(pos, singular_depth, ply, singular_beta - 1, singular_beta, cut_node);
                ctx_->node_stack[ply_idx].excluded_move = Move::null();

                if (is_stopped()) return 0;

                if (se_score < singular_beta) {
                    if (!IsPV && se_score < singular_beta - std::max(25, 3 * depth)) {
                        extension = 2;
                    }
                    else {
                        extension = 1;
                    }
                    ++stats_.singular_extends;
                }
                else if (singular_beta >= beta) {
                    // Multi-Cut: 连排除 tt_move 后的次优走法都能突破 beta
                    return singular_beta;
                }
                else if (tt_score >= beta) {
                    // 负延伸 (Negative Extension): 次优走法也能突破，局面平庸非单向独大，扣减 1 层节约算力
                    extension = -1;
                }
            }

            // 在执行走子前快速判定该步是否不亏子（好吃子阶段免测直接放行）
            const bool is_see_good = (stage == PickerStage::GoodCaptures) || see_ge(pos, m, 0);

            // 执行走子并增量更新子层 NNUE 累加器
            const Color us = pos.side_to_move();
            const Piece moved = pos.piece_on(m.from_sq());
            StateInfo st{};
            pos.make_move(m, st);

            const nnue::DirtyFeatures df = nnue::extract_dirty_features(us, moved, m, st);
            evaluator_->update_on_make_move(pos, df, ctx_->acc_stack[ply_idx], ctx_->acc_stack[ply_idx + 1], &ctx_->finny_table);

            ctx_->node_stack[ply_idx].current_move = m;
            ctx_->node_stack[ply_idx].moved_piece = moved;

            const bool gives_check = in_check(pos);
            if (extension == 0 && gives_check && depth <= 6 && is_see_good) {
                extension = 1;
            }

            // (D2) Passed Pawn Push Extension (仅对真·通路兵逼近第 6/7 横线强力延伸)
            if (extension == 0 && piece_type(moved) == PieceType::Pawn) {
                const uint8_t to_r = rank_of(m.to_sq());
                if (((us == Color::White && to_r >= 5U) || (us == Color::Black && to_r <= 2U)) &&
                    is_passed_pawn(pos, m.to_sq(), us)) {
                    extension = 1;
                }
            }

            const int new_depth = depth - 1 + extension;
            int score = 0;

            // (E) PVS + LMR (Late Move Reductions / 晚步对数减深)
            if (legal_moves == 1) {
                // 第一个走法使用全窗口搜索
                score = -pvs<IsPV>(pos, new_depth, ply + 1, -beta, -alpha, false);
            } else {
                int reduction = 0;
                if (depth >= 3 && legal_moves >= (IsPV ? 4 : 2) && (is_quiet || stage == PickerStage::BadCaptures)) {
                    reduction = kLmrTable.get(depth, legal_moves);
                    if (IsPV) --reduction;
                    if (gives_check) --reduction;
                    if (stage == PickerStage::Killer1 || stage == PickerStage::Killer2 || stage == PickerStage::CounterMove) {
                        --reduction;
                    }
                    if (!improving) ++reduction;
                    if (cut_node)   ++reduction;

                    // 1. 吃子走法（如劣吃子 BadCaptures）限制最大减深幅度不超过 2 层，防范弃子杀局被漏算
                    if (m.is_capture()) {
                        reduction = std::min(reduction, 2);
                    }

                    // 2. 结合主历史与 1-ply / 2-ply 延续历史共同微调减深幅度
                    if (is_quiet) {
                        int32_t total_hist = ctx_->heuristics.get_history(us, m);
                        if (prev_piece != Piece::None && prev_to != Square::None) {
                            total_hist += ctx_->heuristics.get_conthist(prev_piece, prev_to, moved, m.to_sq()) * 2;
                        }
                        if (prev2_piece != Piece::None && prev2_to != Square::None) {
                            total_hist += ctx_->heuristics.get_conthist2(prev2_piece, prev2_to, moved, m.to_sq());
                        }
                        reduction -= total_hist / 4096;
                        // 历史分分级严惩：对战术恶劣走法追加 1~2 层深度减深
                        //if (total_hist < -4000) ++reduction;
                        //if (total_hist < -8000) ++reduction;
                        //if (total_hist > 4000 && reduction > 0) --reduction;
                    }

                    // 3. 劣质吃子（Bad Captures）加重减深
                    if (stage == PickerStage::BadCaptures) {
                        ++reduction;
                    }

                    // 限制安静走法最大减深幅度，防止浅中层关键安静步直接跌出主搜索视野
                    if (is_quiet) {
                        reduction = std::min(reduction, depth / 2 + 1);
                    }

                    // 威胁升变的过河强力兵推进豁免减深
                    const bool is_dangerous_pawn = (piece_type(moved) == PieceType::Pawn &&
                        ((us == Color::White && rank_of(m.to_sq()) >= 5U) ||
                         (us == Color::Black && rank_of(m.to_sq()) <= 2U)));
                    if (is_dangerous_pawn) {
                        reduction = 0;
                    }

                    reduction = std::clamp(reduction, 0, std::max(0, new_depth - 1));
                }

                // 1) 减深零宽窗口探测 (LMR + Zero-Window)
                score = -pvs<false>(pos, new_depth - reduction, ply + 1, -alpha - 1, -alpha, true);

                // 2) 若减深搜索突破 alpha，必须消除减深幻觉：在正常完整深度 new_depth 下重搜！
                if (reduction > 0 && score > alpha) {
                    ++stats_.lmr_researches;
                    score = -pvs<false>(pos, new_depth, ply + 1, -alpha - 1, -alpha, !cut_node);
                }

                // 3) 若在 PV 节点上零宽初搜依然突破 alpha（且未超过 beta 发生截断），展开全窗口重搜
                if constexpr (IsPV) {
                    if (score > alpha && score < beta) {
                        score = -pvs<true>(pos, new_depth, ply + 1, -beta, -alpha, false);
                    }
                }
            }

            pos.undo_move(m, st);

            if (is_stopped()) return 0;

            if (is_quiet && num_searched_quiets < searched_quiets.size()) {
                searched_quiets[num_searched_quiets++] = m;
            }

            if (score > best_score) {
                best_score = score;
                best_move  = m;

                if (score > alpha) {
                    alpha = score;
                    if constexpr (IsPV) {
                        update_pv(ply, m);
                    }

                    if (score >= beta) {
                        // Beta 截断：更新 Killer、Countermove 与 History 表（含对之前失败安静走法的负反馈惩罚）
                        if (is_quiet) {
                            ctx_->heuristics.update_killer(ply, m);
                            if (ply > 0) {
                                ctx_->heuristics.update_countermove(prev_piece, prev_to, m);
                            }
                            const int bonus = std::min(2400, 350 * depth - 200);
                            ctx_->heuristics.update_history(us, m, bonus);
                            if (prev_piece != Piece::None && prev_to != Square::None) {
                                ctx_->heuristics.update_conthist(prev_piece, prev_to, moved, m.to_sq(), bonus);
                            }
                            if (prev2_piece != Piece::None && prev2_to != Square::None) {
                                ctx_->heuristics.update_conthist2(prev2_piece, prev2_to, moved, m.to_sq(), bonus);
                            }
                            for (size_t q = 0; q + 1 < num_searched_quiets; ++q) {
                                ctx_->heuristics.update_history(us, searched_quiets[q], -bonus);
                                const Piece prev_q_moved = pos.piece_on(searched_quiets[q].from_sq());
                                if (prev_piece != Piece::None && prev_to != Square::None) {
                                    ctx_->heuristics.update_conthist(prev_piece, prev_to, prev_q_moved, searched_quiets[q].to_sq(), -bonus);
                                }
                                if (prev2_piece != Piece::None && prev2_to != Square::None) {
                                    ctx_->heuristics.update_conthist2(prev2_piece, prev2_to, prev_q_moved, searched_quiets[q].to_sq(), -bonus);
                                }
                            }
                        }
                        else if (m.is_capture()) {
                            // ▼▼▼ 新增部分：若引发截断的是吃子，记录到吃子历史表中 ▼▼▼
                            const Piece victim_p = m.is_en_passant()
                                ? make_piece(~us, PieceType::Pawn)
                                : pos.piece_on(m.to_sq());
                            const int bonus = std::min(2400, 350 * depth - 200);
                            ctx_->heuristics.update_capture_history(moved, m.to_sq(), piece_type(victim_p), bonus);
                        }
                        break;
                    }
                }
            }
        }

        // 6. 将死 (Checkmate) 与 逼和 (Stalemate) 判定
        if (legal_moves == 0) {
            if (!excluded_move.is_none()) {
                return alpha;
            }
            return is_in_check ? (-kMateScore + ply) : draw_score(pos, ply);
        }

        // 7. 写入置换表 (奇异搜索时不污染主置换表)
        if (excluded_move.is_none()) {
            const TTBound bound = (best_score >= beta)
                ? TTBound::LowerBound
                : (best_score > orig_alpha ? TTBound::Exact : TTBound::UpperBound);
            tt_->store(pos.zobrist_key(), best_move, best_score, raw_eval, depth, bound, ply);

            // 更新多源 CorrHist 评估校准历史
            if (!is_in_check && static_eval != kValueNone && !is_mate_score(best_score)) {
                if (bound == TTBound::Exact ||
                    (bound == TTBound::LowerBound && best_score >= beta) ||
                    (bound == TTBound::UpperBound && best_score <= orig_alpha))
                {
                    const int bonus = std::clamp((best_score - static_eval) * 16, -512, 512);
                    ctx_->heuristics.update_correction(
                        pos.side_to_move(),
                        compute_pawn_key(pos),
                        compute_non_pawn_key(pos, Color::White),
                        compute_non_pawn_key(pos, Color::Black),
                        depth,
                        bonus);
                }
            }
        }

        return best_score;
    }
};

} // namespace valkyrie::search

// ============================================================================
// 9. 阶段四 PVS 搜索与剪枝自检入口 (当定义 VALKYRIE_STAGE4_MAIN 时编译为独立可执行文件)
//    g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -Werror -x c++ -DVALKYRIE_STAGE4_MAIN search.hpp -o search_test
// ============================================================================
#ifdef VALKYRIE_STAGE4_MAIN
int main() {
    using namespace valkyrie;

    std::cout << "=== Valkyrie C++20 Engine (search.hpp) - Stage 4 TT, MovePicker & PVS Self-Test ===\n";

    // 1. 验证 Mate Score <-> Ply 转化恒等式
    constexpr int kTestMateIn3 = search::kMateScore - 5; // ply = 5 处的杀棋分
    constexpr int kStoredTT    = search::score_to_tt(kTestMateIn3, 5);
    constexpr int kRestoredPly8 = search::score_from_tt(kStoredTT, 8);
    static_assert(kStoredTT == search::kMateScore, "TT Mate normalization failed");
    static_assert(kRestoredPly8 == search::kMateScore - 8, "TT Mate ply restoration failed");
    std::cout << "[PASS] Mate Score <-> Ply Normalization : Verified\n";

    // 2. 初始化 16MB 置换表与 Searcher
    search::Searcher searcher(16);
    std::cout << "[PASS] Transposition Table Allocated    : " << searcher.tt().size_mb()
              << " MB (" << searcher.tt().size_entries() << " entries, 16B/entry)\n";

    // 3. 测试战术杀棋发现 (Scholar's Mate in 1: Qxf7#)
    Position mate_pos;
    const bool ok1 = mate_pos.set_from_fen("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");
    assert(ok1);

    search::SearchLimits mate_limits{};
    mate_limits.max_depth = 4;
    const auto mate_stats = searcher.search_position(mate_pos, mate_limits);
    std::cout << "[TEST] Mate-in-1 Tactical Position      : BestMove = " << mate_stats.best_move.to_uci()
              << " | Score = " << mate_stats.best_score
              << " | Nodes = " << mate_stats.nodes << "\n";
    assert(mate_stats.best_move.to_uci() == "h5f7");
    assert(mate_stats.best_score >= search::kMateThreshold);

    // 4. 测试 Kiwipete 复杂中局深度 6 迭代加深 + 期望窗口 + PVS 剪枝
    Position kiwi_pos;
    const bool ok2 = kiwi_pos.set_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    assert(ok2);

    search::SearchLimits kiwi_limits{};
    kiwi_limits.max_depth    = 6;
    kiwi_limits.hard_time_ms = 5000;
    const auto kiwi_stats = searcher.search_position(kiwi_pos, kiwi_limits);

    std::cout << "[TEST] Kiwipete Iterative Deepening d=6 : BestMove = " << kiwi_stats.best_move.to_uci()
              << " | Score = " << kiwi_stats.best_score << " cp"
              << " | Depth = " << kiwi_stats.depth << "/" << kiwi_stats.seldepth
              << " | Nodes = " << kiwi_stats.nodes
              << " | TT Hits = " << kiwi_stats.tt_hits
              << " | NPS = " << kiwi_stats.nps << "\n";
    std::cout << "       Pruning Telemetry -> RFP: " << kiwi_stats.rfp_prunes
              << ", NMP: " << kiwi_stats.nmp_prunes
              << ", LMP: " << kiwi_stats.lmp_prunes
              << ", LMR Researches: " << kiwi_stats.lmr_researches << "\n";
    std::cout << "       PV Line: ";
    for (Move m : kiwi_stats.pv_line) {
        std::cout << m.to_uci() << " ";
    }
    std::cout << "\n";

    assert(!kiwi_stats.best_move.is_none());
    assert(kiwi_stats.nodes > 0);

    std::cout << "=== All Stage 4 Search & Pruning Assertions Passed! ===\n";
    return 0;
}
#endif
