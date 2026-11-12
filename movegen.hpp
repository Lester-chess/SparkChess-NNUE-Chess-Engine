/**
 * @file movegen.hpp
 * @brief Valkyrie C++20 Chess Engine - Stage 2: Precomputed Attack Tables,
 *        Fancy Magic Bitboards (with optional BMI2 PEXT hardware acceleration),
 *        Pseudo-Legal Move Generator, Pin/En-Passant-Aware Legal Check, and Perft Suite.
 * @standard C++20 (-std=c++20 / /std:c++20)
 * @license MIT (100% Original Architecture designed for HalfKAv2 512x2 NNUE)
 */

#pragma once

#include "position.hpp"

#include <chrono>

#if defined(__BMI2__) && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
#include <immintrin.h>
#define VALKYRIE_USE_PEXT 1
#else
#define VALKYRIE_USE_PEXT 0
#endif

namespace valkyrie {

// ============================================================================
// 1. 预计算非滑动棋子攻击表 (兵、马、王：编译期 consteval 零开销生成)
// ============================================================================

namespace attacks {

struct LeaperTables {
    std::array<std::array<Bitboard, 64>, 2> pawn{};   // [Color][Square]
    std::array<Bitboard, 64>                knight{}; // [Square]
    std::array<Bitboard, 64>                king{};   // [Square]
};

[[nodiscard]] consteval LeaperTables generate_leaper_tables() noexcept {
    LeaperTables tbl{};
    for (uint8_t s = 0; s < 64U; ++s) {
        const Square sq = static_cast<Square>(s);
        const Bitboard b = Bitboard::from_square(sq);

        // 白兵攻击 (NorthEast | NorthWest) & 黑兵攻击 (SouthEast | SouthWest)
        tbl.pawn[static_cast<size_t>(Color::White)][s] =
            b.shift<Direction::NorthEast>() | b.shift<Direction::NorthWest>();
        tbl.pawn[static_cast<size_t>(Color::Black)][s] =
            b.shift<Direction::SouthEast>() | b.shift<Direction::SouthWest>();

        // 马的 8 个跳跃方向
        uint64_t n_bits = 0ULL;
        const uint64_t raw = b.raw();
        n_bits |= (raw & Bitboard::kNotFileH)  << 17U; // NNE
        n_bits |= (raw & Bitboard::kNotFileGH) << 10U; // ENE
        n_bits |= (raw & Bitboard::kNotFileGH) >>  6U; // ESE
        n_bits |= (raw & Bitboard::kNotFileH)  >> 15U; // SSE
        n_bits |= (raw & Bitboard::kNotFileA)  << 15U; // NNW
        n_bits |= (raw & Bitboard::kNotFileAB) <<  6U; // WNW
        n_bits |= (raw & Bitboard::kNotFileAB) >> 10U; // WSW
        n_bits |= (raw & Bitboard::kNotFileA)  >> 17U; // SSW
        tbl.knight[s] = Bitboard(n_bits);

        // 王的 8 个相邻方向
        const Bitboard ew = b | b.shift<Direction::East>() | b.shift<Direction::West>();
        const Bitboard k_bb = ew | ew.shift<Direction::North>() | ew.shift<Direction::South>();
        tbl.king[s] = k_bb & ~b;
    }
    return tbl;
}

inline constexpr LeaperTables kLeapers = generate_leaper_tables();

[[nodiscard]] constexpr Bitboard pawn_attacks(Color c, Square sq) noexcept {
    assert(c != Color::None && sq_index(sq) < 64U);
    return kLeapers.pawn[static_cast<size_t>(c)][sq_index(sq)];
}

[[nodiscard]] constexpr Bitboard knight_attacks(Square sq) noexcept {
    assert(sq_index(sq) < 64U);
    return kLeapers.knight[sq_index(sq)];
}

[[nodiscard]] constexpr Bitboard king_attacks(Square sq) noexcept {
    assert(sq_index(sq) < 64U);
    return kLeapers.king[sq_index(sq)];
}

// ============================================================================
// 2. 滑动棋子攻击表 (Fancy Magic Bitboards + BMI2 PEXT 硬件加速双模支持)
// ============================================================================

// 辅助函数：在给定阻挡位棋盘 occ 下，按射线方向计算象或车的真实攻击格
[[nodiscard]] constexpr Bitboard sliding_attacks_on_the_fly(
    Square sq, Bitboard occ, bool is_bishop) noexcept
{
    Bitboard result(0ULL);
    const int r0 = static_cast<int>(rank_of(sq));
    const int f0 = static_cast<int>(file_of(sq));

    constexpr int kRookDirs[4][2]   = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int kBishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const auto& dirs = is_bishop ? kBishopDirs : kRookDirs;

    for (int d = 0; d < 4; ++d) {
        int r = r0 + dirs[d][0];
        int f = f0 + dirs[d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            const Square target = make_square(static_cast<uint8_t>(f), static_cast<uint8_t>(r));
            result.set(target);
            if (occ.test(target)) break; // 遇到阻挡子停止射线延伸
            r += dirs[d][0];
            f += dirs[d][1];
        }
    }
    return result;
}

// 计算某格象或车的内部相关阻挡掩码 (剔除棋盘边缘格)
[[nodiscard]] constexpr Bitboard compute_relevant_mask(Square sq, bool is_bishop) noexcept {
    Bitboard mask(0ULL);
    const int r0 = static_cast<int>(rank_of(sq));
    const int f0 = static_cast<int>(file_of(sq));

    if (is_bishop) {
        constexpr int kDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        for (int d = 0; d < 4; ++d) {
            int r = r0 + kDirs[d][0];
            int f = f0 + kDirs[d][1];
            while (r >= 1 && r <= 6 && f >= 1 && f <= 6) {
                mask.set(make_square(static_cast<uint8_t>(f), static_cast<uint8_t>(r)));
                r += kDirs[d][0];
                f += kDirs[d][1];
            }
        }
    } else {
        for (int r = r0 + 1; r <= 6; ++r) mask.set(make_square(static_cast<uint8_t>(f0), static_cast<uint8_t>(r)));
        for (int r = r0 - 1; r >= 1; --r) mask.set(make_square(static_cast<uint8_t>(f0), static_cast<uint8_t>(r)));
        for (int f = f0 + 1; f <= 6; ++f) mask.set(make_square(static_cast<uint8_t>(f), static_cast<uint8_t>(r0)));
        for (int f = f0 - 1; f >= 1; --f) mask.set(make_square(static_cast<uint8_t>(f), static_cast<uint8_t>(r0)));
    }
    return mask;
}

struct MagicEntry {
    const Bitboard* attacks = nullptr;
    uint64_t        mask    = 0ULL;
    uint64_t        magic   = 0ULL;
    uint32_t        shift   = 0U;

    [[nodiscard]] inline uint32_t index(Bitboard occ) const noexcept {
#if VALKYRIE_USE_PEXT
        return static_cast<uint32_t>(_pext_u64(occ.raw(), mask));
#else
        return static_cast<uint32_t>(((occ.raw() & mask) * magic) >> shift);
#endif
    }
};

// Fancy Magic Bitboard 管理器：首次构造时使用确定性稀疏 PRNG 自动求解无碰撞 Magic 常数并填充表
// (车总表大小 = 102,400 项，象总表大小 = 5,248 项)
class SliderAttackTables {
public:
    static constexpr size_t kRookTableSize   = 102400;
    static constexpr size_t kBishopTableSize = 5248;

    SliderAttackTables() noexcept {
        init_slider(true,  bishop_magics_, bishop_table_);
        init_slider(false, rook_magics_,   rook_table_);
    }

    [[nodiscard]] inline Bitboard bishop_attacks(Square sq, Bitboard occ) const noexcept {
        assert(sq_index(sq) < 64U);
        const auto& m = bishop_magics_[sq_index(sq)];
        return m.attacks[m.index(occ)];
    }

    [[nodiscard]] inline Bitboard rook_attacks(Square sq, Bitboard occ) const noexcept {
        assert(sq_index(sq) < 64U);
        const auto& m = rook_magics_[sq_index(sq)];
        return m.attacks[m.index(occ)];
    }

    [[nodiscard]] inline Bitboard queen_attacks(Square sq, Bitboard occ) const noexcept {
        return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
    }

    [[nodiscard]] const MagicEntry& bishop_entry(Square sq) const noexcept {
        return bishop_magics_[sq_index(sq)];
    }

    [[nodiscard]] const MagicEntry& rook_entry(Square sq) const noexcept {
        return rook_magics_[sq_index(sq)];
    }

private:
    std::array<MagicEntry, 64>              bishop_magics_{};
    std::array<MagicEntry, 64>              rook_magics_{};
    std::array<Bitboard, kBishopTableSize>  bishop_table_{};
    std::array<Bitboard, kRookTableSize>    rook_table_{};

    static constexpr uint64_t next_prng(uint64_t& state) noexcept {
        state ^= state >> 12U;
        state ^= state << 25U;
        state ^= state >> 27U;
        return state * 2685821657736338717ULL;
    }

    static constexpr uint64_t sparse_magic_candidate(uint64_t& state) noexcept {
        return next_prng(state) & next_prng(state) & next_prng(state);
    }

    template <size_t TableSize>
    static void init_slider(
        bool is_bishop,
        std::array<MagicEntry, 64>& entries,
        std::array<Bitboard, TableSize>& table) noexcept
    {
        size_t offset = 0;
        uint64_t rng_state = is_bishop ? 0xB1540B5EE1234567ULL : 0x52004B5EE7654321ULL;

        std::array<Bitboard, 4096> occ_subsets{};
        std::array<Bitboard, 4096> ref_attacks{};
        std::array<uint32_t, 4096> epoch_seen{};
        uint32_t current_epoch = 0;

        for (uint8_t s = 0; s < 64U; ++s) {
            const Square sq = static_cast<Square>(s);
            const Bitboard mask_bb = compute_relevant_mask(sq, is_bishop);
            const uint64_t mask = mask_bb.raw();
            const int bits = mask_bb.popcount();
            const uint32_t shift = static_cast<uint32_t>(64 - bits);
            const size_t num_subsets = 1ULL << bits;

            // 使用 Carry-Rippler 枚举 mask 的全部 2^bits 个阻挡子集
            uint64_t subset = 0ULL;
            for (size_t i = 0; i < num_subsets; ++i) {
                occ_subsets[i] = Bitboard(subset);
                ref_attacks[i] = sliding_attacks_on_the_fly(sq, Bitboard(subset), is_bishop);
                subset = (subset - mask) & mask;
            }

            Bitboard* sq_table = table.data() + offset;
            offset += num_subsets;

#if VALKYRIE_USE_PEXT
            entries[s] = MagicEntry{sq_table, mask, 0ULL, shift};
            for (size_t i = 0; i < num_subsets; ++i) {
                const uint32_t idx = static_cast<uint32_t>(_pext_u64(occ_subsets[i].raw(), mask));
                sq_table[idx] = ref_attacks[i];
            }
#else
            // 搜索无冲突（或良性冲突）的 64位 Magic 乘数
            uint64_t magic = 0ULL;
            bool found = false;
            while (!found) {
                magic = sparse_magic_candidate(rng_state);
                if (std::popcount((mask * magic) & 0xFF00000000000000ULL) < 6) {
                    continue;
                }
                ++current_epoch;
                found = true;
                for (size_t i = 0; i < num_subsets; ++i) {
                    const uint32_t idx = static_cast<uint32_t>((occ_subsets[i].raw() * magic) >> shift);
                    if (epoch_seen[idx] < current_epoch) {
                        epoch_seen[idx] = current_epoch;
                        sq_table[idx]   = ref_attacks[i];
                    } else if (sq_table[idx] != ref_attacks[i]) {
                        found = false;
                        break;
                    }
                }
            }
            entries[s] = MagicEntry{sq_table, mask, magic, shift};
#endif
        }
        assert(offset == TableSize);
    }
};

inline const SliderAttackTables kSliders{};

[[nodiscard]] inline Bitboard bishop_attacks(Square sq, Bitboard occ) noexcept {
    return kSliders.bishop_attacks(sq, occ);
}

[[nodiscard]] inline Bitboard rook_attacks(Square sq, Bitboard occ) noexcept {
    return kSliders.rook_attacks(sq, occ);
}

[[nodiscard]] inline Bitboard queen_attacks(Square sq, Bitboard occ) noexcept {
    return kSliders.queen_attacks(sq, occ);
}

} // namespace attacks

// ============================================================================
// 3. 棋盘格受攻检测与将帅安全性判定
// ============================================================================

// 判断目标格 sq 是否被 attacker_color 方攻击（给定当前自定义占位 occ）
[[nodiscard]] inline bool is_square_attacked(
    const Position& pos, Square sq, Color attacker_color, Bitboard occ) noexcept
{
    assert(sq_index(sq) < 64U && attacker_color != Color::None);

    // 1. 兵攻击：从 sq 向己方兵攻击方向反推是否存在敌方兵
    const Color defender_color = ~attacker_color;
    if ((attacks::pawn_attacks(defender_color, sq) & pos.pieces(attacker_color, PieceType::Pawn)).any()) {
        return true;
    }

    // 2. 马攻击
    if ((attacks::knight_attacks(sq) & pos.pieces(attacker_color, PieceType::Knight)).any()) {
        return true;
    }

    // 3. 王攻击
    if ((attacks::king_attacks(sq) & pos.pieces(attacker_color, PieceType::King)).any()) {
        return true;
    }

    // 4. 象 / 后 斜线攻击
    const Bitboard diag_sliders =
        pos.pieces(attacker_color, PieceType::Bishop) |
        pos.pieces(attacker_color, PieceType::Queen);
    if ((attacks::bishop_attacks(sq, occ) & diag_sliders).any()) {
        return true;
    }

    // 5. 车 / 后 直线攻击
    const Bitboard orth_sliders =
        pos.pieces(attacker_color, PieceType::Rook) |
        pos.pieces(attacker_color, PieceType::Queen);
    if ((attacks::rook_attacks(sq, occ) & orth_sliders).any()) {
        return true;
    }

    return false;
}

[[nodiscard]] inline bool is_square_attacked(
    const Position& pos, Square sq, Color attacker_color) noexcept
{
    return is_square_attacked(pos, sq, attacker_color, pos.occupancy());
}

[[nodiscard]] inline bool in_check(const Position& pos, Color side) noexcept {
    const Bitboard k_bb = pos.pieces(side, PieceType::King);
    if (k_bb.empty()) return false;
    return is_square_attacked(pos, k_bb.lsb(), ~side, pos.occupancy());
}

[[nodiscard]] inline bool in_check(const Position& pos) noexcept {
    return in_check(pos, pos.side_to_move());
}

// ============================================================================
// 4. MoveList 固定容量栈数组容器 (零堆内存分配，最大容纳 256 步走法)
// ============================================================================

class MoveList {
public:
    static constexpr size_t kMaxMoves = 256;

    constexpr MoveList() noexcept = default;

    constexpr void push_back(Move m) noexcept {
        assert(size_ < kMaxMoves);
        moves_[size_++] = m;
    }

    constexpr void emplace_back(Square from, Square to, MoveFlag flags = MoveFlag::Quiet) noexcept {
        assert(size_ < kMaxMoves);
        moves_[size_++] = Move(from, to, flags);
    }

    [[nodiscard]] constexpr size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr void clear() noexcept { size_ = 0; }

    [[nodiscard]] constexpr Move operator[](size_t idx) const noexcept {
        assert(idx < size_);
        return moves_[idx];
    }

    [[nodiscard]] constexpr const Move* begin() const noexcept { return moves_.data(); }
    [[nodiscard]] constexpr const Move* end() const noexcept { return moves_.data() + size_; }
    [[nodiscard]] constexpr Move* begin() noexcept { return moves_.data(); }
    [[nodiscard]] constexpr Move* end() noexcept { return moves_.data() + size_; }

private:
    std::array<Move, kMaxMoves> moves_{};
    size_t                      size_ = 0;
};

// ============================================================================
// 5. 伪合法走法生成器 (generate_pseudo_legal_moves) 与走法合法性校验 (is_legal)
// ============================================================================

inline void push_promotions(MoveList& list, Square from, Square to, bool is_capture) noexcept {
    if (is_capture) {
        list.emplace_back(from, to, MoveFlag::PromoCaptureQueen);
        list.emplace_back(from, to, MoveFlag::PromoCaptureRook);
        list.emplace_back(from, to, MoveFlag::PromoCaptureBishop);
        list.emplace_back(from, to, MoveFlag::PromoCaptureKnight);
    } else {
        list.emplace_back(from, to, MoveFlag::PromoQueen);
        list.emplace_back(from, to, MoveFlag::PromoRook);
        list.emplace_back(from, to, MoveFlag::PromoBishop);
        list.emplace_back(from, to, MoveFlag::PromoKnight);
    }
}

// 生成当前走子方的全部伪合法走法（常规走子、双步推进、吃子、升变、吃过路兵、王车易位）
inline void generate_pseudo_legal_moves(const Position& pos, MoveList& list) noexcept {
    list.clear();

    const Color    us       = pos.side_to_move();
    const Color    them     = ~us;
    const Bitboard our_bb   = pos.pieces(us);
    const Bitboard their_all_bb = pos.pieces(them);
    const Bitboard their_bb     = their_all_bb & ~pos.pieces(them, PieceType::King); // 严禁生成吃王走法
    const Bitboard occ_bb       = our_bb | their_all_bb;
    const Bitboard empty_bb = ~occ_bb;

    // ------------------------------------------------------------------------
    // 1. 兵走法 (Pawns: 单步推进、双步推进、左/右斜吃子、底线升变、吃过路兵)
    // ------------------------------------------------------------------------
    const Bitboard pawns = pos.pieces(us, PieceType::Pawn);
    if (us == Color::White) {
        const Bitboard promo_pawns     = pawns & Bitboard(0x00FF000000000000ULL); // Rank 7
        const Bitboard non_promo_pawns = pawns & ~Bitboard(0x00FF000000000000ULL);

        // 1a. 非升变白兵单步与双步推进
        const Bitboard single_push = non_promo_pawns.shift<Direction::North>() & empty_bb;
        const Bitboard double_push = (single_push & Bitboard(Bitboard::kRank1 << 16U)).shift<Direction::North>() & empty_bb;

        for (Square to : single_push) {
            list.emplace_back(static_cast<Square>(sq_index(to) - 8U), to, MoveFlag::Quiet);
        }
        for (Square to : double_push) {
            list.emplace_back(static_cast<Square>(sq_index(to) - 16U), to, MoveFlag::DoublePawnPush);
        }

        // 1b. 非升变白兵吃子
        const Bitboard cap_nw = non_promo_pawns.shift<Direction::NorthWest>() & their_bb;
        const Bitboard cap_ne = non_promo_pawns.shift<Direction::NorthEast>() & their_bb;
        for (Square to : cap_nw) {
            list.emplace_back(static_cast<Square>(sq_index(to) - 7U), to, MoveFlag::Capture);
        }
        for (Square to : cap_ne) {
            list.emplace_back(static_cast<Square>(sq_index(to) - 9U), to, MoveFlag::Capture);
        }

        // 1c. 第7横排白兵升变（直进升变 + 左/右斜吃升变）
        if (promo_pawns.any()) {
            const Bitboard promo_push = promo_pawns.shift<Direction::North>() & empty_bb;
            const Bitboard promo_nw   = promo_pawns.shift<Direction::NorthWest>() & their_bb;
            const Bitboard promo_ne   = promo_pawns.shift<Direction::NorthEast>() & their_bb;

            for (Square to : promo_push) {
                push_promotions(list, static_cast<Square>(sq_index(to) - 8U), to, false);
            }
            for (Square to : promo_nw) {
                push_promotions(list, static_cast<Square>(sq_index(to) - 7U), to, true);
            }
            for (Square to : promo_ne) {
                push_promotions(list, static_cast<Square>(sq_index(to) - 9U), to, true);
            }
        }

        // 1d. 白兵吃过路兵
        const Square ep_sq = pos.ep_square();
        if (ep_sq != Square::None) {
            const Bitboard ep_attackers = attacks::pawn_attacks(Color::Black, ep_sq) & non_promo_pawns;
            for (Square from : ep_attackers) {
                list.emplace_back(from, ep_sq, MoveFlag::EnPassantCapture);
            }
        }
    } else {
        const Bitboard promo_pawns     = pawns & Bitboard(0x000000000000FF00ULL); // Rank 2
        const Bitboard non_promo_pawns = pawns & ~Bitboard(0x000000000000FF00ULL);

        // 1a. 非升变黑兵单步与双步推进
        const Bitboard single_push = non_promo_pawns.shift<Direction::South>() & empty_bb;
        const Bitboard double_push = (single_push & Bitboard(Bitboard::kRank1 << 40U)).shift<Direction::South>() & empty_bb;

        for (Square to : single_push) {
            list.emplace_back(static_cast<Square>(sq_index(to) + 8U), to, MoveFlag::Quiet);
        }
        for (Square to : double_push) {
            list.emplace_back(static_cast<Square>(sq_index(to) + 16U), to, MoveFlag::DoublePawnPush);
        }

        // 1b. 非升变黑兵吃子
        const Bitboard cap_sw = non_promo_pawns.shift<Direction::SouthWest>() & their_bb;
        const Bitboard cap_se = non_promo_pawns.shift<Direction::SouthEast>() & their_bb;
        for (Square to : cap_sw) {
            list.emplace_back(static_cast<Square>(sq_index(to) + 9U), to, MoveFlag::Capture);
        }
        for (Square to : cap_se) {
            list.emplace_back(static_cast<Square>(sq_index(to) + 7U), to, MoveFlag::Capture);
        }

        // 1c. 第2横排黑兵升变
        if (promo_pawns.any()) {
            const Bitboard promo_push = promo_pawns.shift<Direction::South>() & empty_bb;
            const Bitboard promo_sw   = promo_pawns.shift<Direction::SouthWest>() & their_bb;
            const Bitboard promo_se   = promo_pawns.shift<Direction::SouthEast>() & their_bb;

            for (Square to : promo_push) {
                push_promotions(list, static_cast<Square>(sq_index(to) + 8U), to, false);
            }
            for (Square to : promo_sw) {
                push_promotions(list, static_cast<Square>(sq_index(to) + 9U), to, true);
            }
            for (Square to : promo_se) {
                push_promotions(list, static_cast<Square>(sq_index(to) + 7U), to, true);
            }
        }

        // 1d. 黑兵吃过路兵
        const Square ep_sq = pos.ep_square();
        if (ep_sq != Square::None) {
            const Bitboard ep_attackers = attacks::pawn_attacks(Color::White, ep_sq) & non_promo_pawns;
            for (Square from : ep_attackers) {
                list.emplace_back(from, ep_sq, MoveFlag::EnPassantCapture);
            }
        }
    }

    // ------------------------------------------------------------------------
    // 2. 马走法 (Knights)
    // ------------------------------------------------------------------------
    for (Square from : pos.pieces(us, PieceType::Knight)) {
        const Bitboard targets = attacks::knight_attacks(from) & ~our_bb;
        for (Square to : (targets & their_bb)) {
            list.emplace_back(from, to, MoveFlag::Capture);
        }
        for (Square to : (targets & empty_bb)) {
            list.emplace_back(from, to, MoveFlag::Quiet);
        }
    }

    // ------------------------------------------------------------------------
    // 3. 象走法 (Bishops)
    // ------------------------------------------------------------------------
    for (Square from : pos.pieces(us, PieceType::Bishop)) {
        const Bitboard targets = attacks::bishop_attacks(from, occ_bb) & ~our_bb;
        for (Square to : (targets & their_bb)) {
            list.emplace_back(from, to, MoveFlag::Capture);
        }
        for (Square to : (targets & empty_bb)) {
            list.emplace_back(from, to, MoveFlag::Quiet);
        }
    }

    // ------------------------------------------------------------------------
    // 4. 车走法 (Rooks)
    // ------------------------------------------------------------------------
    for (Square from : pos.pieces(us, PieceType::Rook)) {
        const Bitboard targets = attacks::rook_attacks(from, occ_bb) & ~our_bb;
        for (Square to : (targets & their_bb)) {
            list.emplace_back(from, to, MoveFlag::Capture);
        }
        for (Square to : (targets & empty_bb)) {
            list.emplace_back(from, to, MoveFlag::Quiet);
        }
    }

    // ------------------------------------------------------------------------
    // 5. 后走法 (Queens)
    // ------------------------------------------------------------------------
    for (Square from : pos.pieces(us, PieceType::Queen)) {
        const Bitboard targets = attacks::queen_attacks(from, occ_bb) & ~our_bb;
        for (Square to : (targets & their_bb)) {
            list.emplace_back(from, to, MoveFlag::Capture);
        }
        for (Square to : (targets & empty_bb)) {
            list.emplace_back(from, to, MoveFlag::Quiet);
        }
    }

    // ------------------------------------------------------------------------
    // 6. 王走法与王车易位 (King & Castling)
    // ------------------------------------------------------------------------
    const Bitboard king_bb = pos.pieces(us, PieceType::King);
    if (king_bb.any()) {
        const Square ksq = king_bb.lsb();
        const Bitboard k_targets = attacks::king_attacks(ksq) & ~our_bb;
        for (Square to : (k_targets & their_bb)) {
            list.emplace_back(ksq, to, MoveFlag::Capture);
        }
        for (Square to : (k_targets & empty_bb)) {
            list.emplace_back(ksq, to, MoveFlag::Quiet);
        }

        // 王车易位生成（要求通道无阻挡子，且起点/途经格/终点格在 is_legal 或此处初步筛查）
        const CastlingRights cr = pos.castling_rights();
        if (us == Color::White && ksq == Square::e1) {
            // 白方王翼易位 O-O (e1 -> g1, 需 f1, g1 为空)
            if ((cr & CastlingRights::WhiteKingside) != CastlingRights::None) {
                constexpr Bitboard kPathF1G1((1ULL << sq_index(Square::f1)) | (1ULL << sq_index(Square::g1)));
                if ((occ_bb & kPathF1G1).empty()) {
                    list.emplace_back(Square::e1, Square::g1, MoveFlag::KingCastle);
                }
            }
            // 白方后翼易位 O-O-O (e1 -> c1, 需 b1, c1, d1 为空)
            if ((cr & CastlingRights::WhiteQueenside) != CastlingRights::None) {
                constexpr Bitboard kPathB1C1D1(
                    (1ULL << sq_index(Square::b1)) |
                    (1ULL << sq_index(Square::c1)) |
                    (1ULL << sq_index(Square::d1)));
                if ((occ_bb & kPathB1C1D1).empty()) {
                    list.emplace_back(Square::e1, Square::c1, MoveFlag::QueenCastle);
                }
            }
        } else if (us == Color::Black && ksq == Square::e8) {
            // 黑方王翼易位 O-O (e8 -> g8, 需 f8, g8 为空)
            if ((cr & CastlingRights::BlackKingside) != CastlingRights::None) {
                constexpr Bitboard kPathF8G8((1ULL << sq_index(Square::f8)) | (1ULL << sq_index(Square::g8)));
                if ((occ_bb & kPathF8G8).empty()) {
                    list.emplace_back(Square::e8, Square::g8, MoveFlag::KingCastle);
                }
            }
            // 黑方后翼易位 O-O-O (e8 -> c8, 需 b8, c8, d8 为空)
            if ((cr & CastlingRights::BlackQueenside) != CastlingRights::None) {
                constexpr Bitboard kPathB8C8D8(
                    (1ULL << sq_index(Square::b8)) |
                    (1ULL << sq_index(Square::c8)) |
                    (1ULL << sq_index(Square::d8)));
                if ((occ_bb & kPathB8C8D8).empty()) {
                    list.emplace_back(Square::e8, Square::c8, MoveFlag::QueenCastle);
                }
            }
        }
    }
}

// 严格合法性检验 is_legal(pos, move)：无需拷贝整个 Position，直接基于位运算与射线阻挡检验走后己方王是否受攻
[[nodiscard]] inline bool is_legal(const Position& pos, Move m) noexcept {
    if (m.is_none()) return false;
    // 严禁吃王（国际象棋基本规则防溢出屏障）
    if (pos.piece_on(m.to_sq()) == make_piece(~pos.side_to_move(), PieceType::King)) return false;

    const Color    us     = pos.side_to_move();
    const Color    them   = ~us;
    const Square   from   = m.from_sq();
    const Square   to     = m.to_sq();
    const MoveFlag fl     = m.flags();
    const Bitboard occ_bb = pos.occupancy();

    const Bitboard king_bb = pos.pieces(us, PieceType::King);
    if (king_bb.empty()) return false;
    const Square ksq = king_bb.lsb();

    // 1. 王车易位合法性：王不能正被将军，且途经格与目标格均不能受敌方攻击
    if (fl == MoveFlag::KingCastle) {
        const Square pass_sq = (us == Color::White) ? Square::f1 : Square::f8;
        const Square dest_sq = (us == Color::White) ? Square::g1 : Square::g8;
        if (is_square_attacked(pos, ksq,     them, occ_bb)) return false;
        if (is_square_attacked(pos, pass_sq, them, occ_bb)) return false;
        if (is_square_attacked(pos, dest_sq, them, occ_bb)) return false;
        return true;
    }
    if (fl == MoveFlag::QueenCastle) {
        const Square pass_sq = (us == Color::White) ? Square::d1 : Square::d8;
        const Square dest_sq = (us == Color::White) ? Square::c1 : Square::c8;
        if (is_square_attacked(pos, ksq,     them, occ_bb)) return false;
        if (is_square_attacked(pos, pass_sq, them, occ_bb)) return false;
        if (is_square_attacked(pos, dest_sq, them, occ_bb)) return false;
        return true;
    }

    // 2. 吃过路兵合法性：模拟移走起跳兵、放置到达格兵并移除被吃敌兵（防止第4/5横排水平双兵消失引发的车/后闪将）
    if (fl == MoveFlag::EnPassantCapture) {
        const Square cap_sq = (us == Color::White)
            ? static_cast<Square>(sq_index(to) - 8U)
            : static_cast<Square>(sq_index(to) + 8U);

        Bitboard occ_after = occ_bb;
        occ_after.clear(from);
        occ_after.clear(cap_sq);
        occ_after.set(to);

        const Bitboard diag_sliders =
            pos.pieces(them, PieceType::Bishop) |
            pos.pieces(them, PieceType::Queen);
        const Bitboard orth_sliders =
            pos.pieces(them, PieceType::Rook) |
            pos.pieces(them, PieceType::Queen);

        if ((attacks::bishop_attacks(ksq, occ_after) & diag_sliders).any()) return false;
        if ((attacks::rook_attacks(ksq, occ_after)   & orth_sliders).any()) return false;
        if ((attacks::knight_attacks(ksq) & pos.pieces(them, PieceType::Knight)).any()) return false;
        if ((attacks::pawn_attacks(us, ksq) & (pos.pieces(them, PieceType::Pawn) & ~Bitboard::from_square(cap_sq))).any()) return false;
        if ((attacks::king_attacks(ksq) & pos.pieces(them, PieceType::King)).any()) return false;
        return true;
    }

    // 3. 王移动合法性：移除原王位阻挡后（防止王顺着车/象攻击射线后退），检查新王格 to 是否被敌方攻击
    if (from == ksq) {
        Bitboard occ_without_king = occ_bb;
        occ_without_king.clear(ksq);
        occ_without_king.set(to);

        // 排除被王吃掉的 to 格敌子
        const Bitboard to_mask = ~Bitboard::from_square(to);

        if ((attacks::pawn_attacks(us, to) & pos.pieces(them, PieceType::Pawn) & to_mask).any()) return false;
        if ((attacks::knight_attacks(to)   & pos.pieces(them, PieceType::Knight) & to_mask).any()) return false;
        if ((attacks::king_attacks(to)     & pos.pieces(them, PieceType::King) & to_mask).any()) return false;

        const Bitboard diag_sliders =
            (pos.pieces(them, PieceType::Bishop) | pos.pieces(them, PieceType::Queen)) & to_mask;
        if ((attacks::bishop_attacks(to, occ_without_king) & diag_sliders).any()) return false;

        const Bitboard orth_sliders =
            (pos.pieces(them, PieceType::Rook) | pos.pieces(them, PieceType::Queen)) & to_mask;
        if ((attacks::rook_attacks(to, occ_without_king) & orth_sliders).any()) return false;

        return true;
    }

    // 4. 其它非王棋子移动：模拟 occ 变化并排除被吃掉的 to 格敌子，检验己方王 ksq 是否暴露在敌方攻击下
    Bitboard occ_after = occ_bb;
    occ_after.clear(from);
    occ_after.set(to);
    const Bitboard to_mask = ~Bitboard::from_square(to);

    const Bitboard diag_sliders =
        (pos.pieces(them, PieceType::Bishop) | pos.pieces(them, PieceType::Queen)) & to_mask;
    if ((attacks::bishop_attacks(ksq, occ_after) & diag_sliders).any()) return false;

    const Bitboard orth_sliders =
        (pos.pieces(them, PieceType::Rook) | pos.pieces(them, PieceType::Queen)) & to_mask;
    if ((attacks::rook_attacks(ksq, occ_after) & orth_sliders).any()) return false;

    if ((attacks::knight_attacks(ksq) & pos.pieces(them, PieceType::Knight) & to_mask).any()) return false;
    if ((attacks::pawn_attacks(us, ksq) & pos.pieces(them, PieceType::Pawn) & to_mask).any()) return false;

    return true;
}

// 便捷函数：直接生成全部严格合法走法
inline void generate_legal_moves(const Position& pos, MoveList& legal_list) noexcept {
    MoveList pseudo;
    generate_pseudo_legal_moves(pos, pseudo);
    legal_list.clear();
    for (Move m : pseudo) {
        if (is_legal(pos, m)) {
            legal_list.push_back(m);
        }
    }
}

// ============================================================================
// 6. Perft 走法枚举性能与规则正确性基准测试器 (支持 Divide 根节点分支明细统计)
// ============================================================================

[[nodiscard]] inline uint64_t perft(Position& pos, int depth) {
    if (depth <= 0) return 1ULL;

    MoveList moves;
    generate_pseudo_legal_moves(pos, moves);

    if (depth == 1) {
        uint64_t count = 0ULL;
        for (Move m : moves) {
            if (is_legal(pos, m)) {
                ++count;
            }
        }
        return count;
    }

    uint64_t nodes = 0ULL;
    StateInfo st{};
    for (Move m : moves) {
        if (!is_legal(pos, m)) continue;
        pos.make_move(m, st);
        nodes += perft(pos, depth - 1);
        pos.undo_move(m, st);
    }
    return nodes;
}

struct PerftDivideEntry {
    Move     move{};
    uint64_t nodes = 0ULL;
};

[[nodiscard]] inline std::vector<PerftDivideEntry> perft_divide(Position& pos, int depth) {
    std::vector<PerftDivideEntry> result;
    if (depth <= 0) return result;

    MoveList moves;
    generate_pseudo_legal_moves(pos, moves);
    result.reserve(moves.size());

    StateInfo st{};
    for (Move m : moves) {
        if (!is_legal(pos, m)) continue;
        pos.make_move(m, st);
        const uint64_t branch_nodes = (depth == 1) ? 1ULL : perft(pos, depth - 1);
        pos.undo_move(m, st);
        result.push_back(PerftDivideEntry{m, branch_nodes});
    }
    return result;
}

} // namespace valkyrie

// ============================================================================
// 7. 阶段二 Perft 标准测试套件入口 (当定义 VALKYRIE_STAGE2_MAIN 时编译为独立测试程序)
//    g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -Werror -x c++ -DVALKYRIE_STAGE2_MAIN movegen.hpp -o perft_test
// ============================================================================
#ifdef VALKYRIE_STAGE2_MAIN
int main() {
    using namespace valkyrie;

    struct PerftCase {
        std::string_view name;
        std::string_view fen;
        int              depth;
        uint64_t         expected_nodes;
    };

    constexpr std::array<PerftCase, 7> kSuite = {{
        {"Startpos (Depth 1)", Position::kStartFEN, 1, 20ULL},
        {"Startpos (Depth 2)", Position::kStartFEN, 2, 400ULL},
        {"Startpos (Depth 3)", Position::kStartFEN, 3, 8902ULL},
        {"Startpos (Depth 4)", Position::kStartFEN, 4, 197281ULL},
        {"Startpos (Depth 5)", Position::kStartFEN, 5, 4865609ULL},
        {"Kiwipete (Depth 4)", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603ULL},
        {"Position 3 EP/Pin (Depth 5)", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624ULL}
    }};

    std::cout << "=== Valkyrie C++20 Engine (movegen.hpp) - Stage 2 Perft Verification Suite ===\n";
    Position pos;
    for (const auto& tc : kSuite) {
        const bool ok = pos.set_from_fen(tc.fen);
        assert(ok);

        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t nodes = perft(pos, tc.depth);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double mnps = (ms > 0.0) ? (static_cast<double>(nodes) / (ms * 1000.0)) : 0.0;

        std::cout << std::left << std::setw(28) << tc.name
                  << " | nodes = " << std::right << std::setw(9) << nodes
                  << " | expected = " << std::setw(9) << tc.expected_nodes
                  << " | " << std::fixed << std::setprecision(2) << std::setw(7) << ms << " ms ("
                  << mnps << " MNPS) "
                  << (nodes == tc.expected_nodes ? "[PASS]" : "[FAIL]") << "\n";
        assert(nodes == tc.expected_nodes);
    }

    std::cout << "All Perft tests passed with 100% accuracy.\n";
    return 0;
}
#endif
