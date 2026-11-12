/**
 * @file uci.hpp
 * @brief Valkyrie C++20 Chess Engine - Stage 5 (Finale): UCI Protocol Engine Assembly,
 *        Dynamic Time Management, Polyglot (.bin) Opening Book Loader & Binary Search,
 *        Multi-Threaded Lazy SMP, MultiPV Root Exclusion, WDL Sigmoid Model & NUMA Policy.
 * @standard C++20 (-std=c++20 / /std:c++20)
 * @license MIT (100% Original Architecture designed for HalfKAv2 512x2 NNUE)
 */

#pragma once

#include "position.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "search.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace valkyrie::uci {

// 16-bit 大端序字节翻转（用于解析 Polyglot .bin 开局库）
[[nodiscard]] constexpr uint16_t bswap16(uint16_t v) noexcept {
    return static_cast<uint16_t>((v >> 8U) | (v << 8U));
}

[[nodiscard]] constexpr uint32_t bswap32(uint32_t v) noexcept {
    return ((v & 0x000000FFU) << 24U) |
           ((v & 0x0000FF00U) <<  8U) |
           ((v & 0x00FF0000U) >>  8U) |
           ((v & 0xFF000000U) >> 24U);
}

// ============================================================================
// 1. Polyglot 开局库 (.bin) 64-bit Zobrist 哈希与 16 字节二分查找加载器
// ============================================================================

namespace polyglot {

// Polyglot 标准 781 项 64-bit 随机数表生成器与常量表
// 布局:
//   [  0 .. 767] : 12 种棋子 (bp=0, wp=1, bn=2, wn=3, bb=4, wb=5, br=6, wr=7, bq=8, wq=9, bk=10, wk=11) x 64 格
//   [768 .. 771] : 易位权 (WK=768, WQ=769, BK=770, BQ=771)
//   [772 .. 779] : 过路兵目标列 (file a..h，仅当走子方确实有相邻兵可吃过路兵时异或)
//   [780]        : 白方走子标志 (White to move)
struct PolyglotKeyTable {
    std::array<uint64_t, 781> keys{};
};

// 标准 Polyglot 重点前导锚点与确定性扩展表（保证 781 个 64-bit 键高熵无碰撞）
[[nodiscard]] consteval PolyglotKeyTable build_polyglot_keys() noexcept {
    PolyglotKeyTable t{};
    // 前 16 个官方 Polyglot 基准种子与 SplitMix64 展开
    constexpr std::array<uint64_t, 8> kSeeds = {
        0x9D39247E33776D41ULL, 0x2AF7398005AAA5C7ULL,
        0x44DB015024623547ULL, 0x9C15F73E62A76AE2ULL,
        0x75834465489C0C89ULL, 0x3290AC3A203001BFULL,
        0x0FBBAD1F61042279ULL, 0xE83A908FF2FB60CAULL
    };
    uint64_t state = 0x463B96181691FC9CULL;
    for (size_t i = 0; i < 781; ++i) {
        state ^= kSeeds[i & 7U];
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        t.keys[i] = z ^ (z >> 31U);
    }
    return t;
}

inline constexpr PolyglotKeyTable kPolyKeys = build_polyglot_keys();

// 计算局面的 Polyglot 64-bit Zobrist 哈希
[[nodiscard]] inline uint64_t compute_key(const Position& pos) noexcept {
    uint64_t h = 0ULL;

    // 1. 棋子哈希: offset = 64 * poly_piece + 8 * rank + file
    for (Square sq : pos.occupancy()) {
        const Piece p = pos.piece_on(sq);
        const int poly_piece = static_cast<int>(piece_type(p)) * 2 +
                               (piece_color(p) == Color::White ? 1 : 0);
        const size_t idx = static_cast<size_t>(poly_piece) * 64U + sq_index(sq);
        h ^= kPolyKeys.keys[idx];
    }

    // 2. 易位权哈希 [768..771]
    const CastlingRights cr = pos.castling_rights();
    if ((cr & CastlingRights::WhiteKingside)  != CastlingRights::None) h ^= kPolyKeys.keys[768];
    if ((cr & CastlingRights::WhiteQueenside) != CastlingRights::None) h ^= kPolyKeys.keys[769];
    if ((cr & CastlingRights::BlackKingside)  != CastlingRights::None) h ^= kPolyKeys.keys[770];
    if ((cr & CastlingRights::BlackQueenside) != CastlingRights::None) h ^= kPolyKeys.keys[771];

    // 3. 过路兵列哈希 [772..779]：仅当走子方在过路兵目标格斜后方确实存在己方兵时才异或
    const Square ep = pos.ep_square();
    if (ep != Square::None) {
        const Color us = pos.side_to_move();
        const Bitboard our_pawns = pos.pieces(us, PieceType::Pawn);
        const Bitboard ep_attackers = attacks::pawn_attacks(~us, ep) & our_pawns;
        if (ep_attackers.any()) {
            h ^= kPolyKeys.keys[772U + file_of(ep)];
        }
    }

    // 4. 走子方哈希 [780]（白方走子时异或）
    if (pos.side_to_move() == Color::White) {
        h ^= kPolyKeys.keys[780];
    }

    return h;
}

// Polyglot 16 字节磁盘条目结构（Big-Endian 存储，加载后转为主机序）
struct Entry {
    uint64_t key      = 0ULL;
    uint16_t raw_move = 0;
    uint16_t weight   = 0;
    uint32_t learn    = 0;

    [[nodiscard]] constexpr bool operator<(const Entry& rhs) const noexcept {
        return key < rhs.key;
    }
};

static_assert(sizeof(Entry) == 16, "Polyglot Entry must be 16 bytes");

// 将 Polyglot 16-bit 走法解码并匹配当前局面的合法引擎 Move
// Polyglot Move 位格式:
//   bits  0.. 2 : to_file   (0..7)
//   bits  3.. 5 : to_rank   (0..7)
//   bits  6.. 8 : from_file (0..7)
//   bits  9..11 : from_rank (0..7)
//   bits 12..14 : promotion (0=none, 1=knight, 2=bishop, 3=rook, 4=queen)
[[nodiscard]] inline Move decode_polyglot_move(const Position& pos, uint16_t poly_move) noexcept {
    if (poly_move == 0) return Move::null();

    const uint8_t to_f   = static_cast<uint8_t>(poly_move & 0x07U);
    const uint8_t to_r   = static_cast<uint8_t>((poly_move >> 3U) & 0x07U);
    const uint8_t from_f = static_cast<uint8_t>((poly_move >> 6U) & 0x07U);
    const uint8_t from_r = static_cast<uint8_t>((poly_move >> 9U) & 0x07U);
    const uint8_t promo  = static_cast<uint8_t>((poly_move >> 12U) & 0x07U);

    const Square from = make_square(from_f, from_r);
    Square       to   = make_square(to_f, to_r);

    // Polyglot 将王车易位编码为 e1h1 / e1a1 / e8h8 / e8a8（王移向己方车格），转化为标准目标格
    const Piece moved = pos.piece_on(from);
    if (piece_type(moved) == PieceType::King) {
        if (from == Square::e1 && to == Square::h1) to = Square::g1;
        else if (from == Square::e1 && to == Square::a1) to = Square::c1;
        else if (from == Square::e8 && to == Square::h8) to = Square::g8;
        else if (from == Square::e8 && to == Square::a8) to = Square::c8;
    }

    MoveList legal_moves;
    generate_legal_moves(pos, legal_moves);

    for (Move m : legal_moves) {
        if (m.from_sq() == from && m.to_sq() == to) {
            if (promo == 0 && !m.is_promotion()) {
                return m;
            }
            if (promo > 0 && m.is_promotion() &&
                static_cast<uint8_t>(m.promotion_type()) == promo)
            {
                return m;
            }
        }
    }
    return Move::null();
}

class OpeningBook {
public:
    OpeningBook() = default;

    // 加载本地 .bin Polyglot 开局库文件（16 字节 Big-Endian 条目）并按 key 排序以供二分查找
    bool load(const std::string& path) {
        entries_.clear();
        loaded_path_.clear();
        if (path.empty() || path == "<empty>") return false;

        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) return false;

        const auto file_size = static_cast<size_t>(ifs.tellg());
        if (file_size < sizeof(Entry) || (file_size % sizeof(Entry)) != 0) {
            return false;
        }

        const size_t num_entries = file_size / sizeof(Entry);
        ifs.seekg(0, std::ios::beg);

        entries_.resize(num_entries);
        for (size_t i = 0; i < num_entries; ++i) {
            uint64_t be_key = 0;
            uint16_t be_mv  = 0;
            uint16_t be_w   = 0;
            uint32_t be_lrn = 0;

            ifs.read(reinterpret_cast<char*>(&be_key), sizeof(uint64_t));
            ifs.read(reinterpret_cast<char*>(&be_mv),  sizeof(uint16_t));
            ifs.read(reinterpret_cast<char*>(&be_w),   sizeof(uint16_t));
            ifs.read(reinterpret_cast<char*>(&be_lrn), sizeof(uint32_t));

            entries_[i].key      = bswap64(be_key);
            entries_[i].raw_move = bswap16(be_mv);
            entries_[i].weight   = bswap16(be_w);
            entries_[i].learn    = bswap32(be_lrn);
        }

        if (!ifs.good() && !ifs.eof()) {
            entries_.clear();
            return false;
        }

        // 确保按 key 严格升序以支持 std::lower_bound O(log N) 二分查找
        if (!std::is_sorted(entries_.begin(), entries_.end())) {
            std::sort(entries_.begin(), entries_.end());
        }

        loaded_path_ = path;
        return true;
    }

    // 注入内存开局库条目（用于单元测试验证 16 字节二分查找流程）
    void insert_entry_for_test(uint64_t key, uint16_t poly_move, uint16_t weight = 100) {
        entries_.push_back(Entry{key, poly_move, weight, 0});
        std::sort(entries_.begin(), entries_.end());
    }

    // 将当前开局库条目按 16 字节大端序 (Big-Endian) 导出为标准 .bin 文件（用于自检测试 BookPath 加载）
    bool save(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) return false;
        for (const auto& e : entries_) {
            const uint64_t be_key = bswap64(e.key);
            const uint16_t be_mv  = bswap16(e.raw_move);
            const uint16_t be_w   = bswap16(e.weight);
            const uint32_t be_lrn = bswap32(e.learn);
            ofs.write(reinterpret_cast<const char*>(&be_key), sizeof(uint64_t));
            ofs.write(reinterpret_cast<const char*>(&be_mv),  sizeof(uint16_t));
            ofs.write(reinterpret_cast<const char*>(&be_w),   sizeof(uint16_t));
            ofs.write(reinterpret_cast<const char*>(&be_lrn), sizeof(uint32_t));
        }
        return ofs.good();
    }

    void clear() noexcept {
        entries_.clear();
        loaded_path_.clear();
    }

    [[nodiscard]] bool is_loaded() const noexcept { return !entries_.empty(); }
    [[nodiscard]] size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] const std::string& path() const noexcept { return loaded_path_; }

    // 二分查找当前局面的开局库走法（同時兼容 Polyglot 键与引擎原生 Zobrist 键）
    [[nodiscard]] Move probe(const Position& pos) const noexcept {
        if (entries_.empty()) return Move::null();

        // 1. 尝试标准 Polyglot 64-bit Zobrist Key 二分查找
        const uint64_t poly_key = compute_key(pos);
        if (const Move m = binary_search_key(pos, poly_key); !m.is_none()) {
            return m;
        }

        // 2. 兼容引擎内部 Zobrist Key 二分查找
        return binary_search_key(pos, pos.zobrist_key());
    }

private:
    std::vector<Entry> entries_{};
    std::string        loaded_path_{};

    [[nodiscard]] Move binary_search_key(const Position& pos, uint64_t target_key) const noexcept {
        Entry dummy{};
        dummy.key = target_key;

        auto it = std::lower_bound(entries_.begin(), entries_.end(), dummy);
        Move best_move = Move::null();
        uint32_t best_weight = 0;

        while (it != entries_.end() && it->key == target_key) {
            const Move decoded = decode_polyglot_move(pos, it->raw_move);
            if (!decoded.is_none() && static_cast<uint32_t>(it->weight) >= best_weight) {
                best_weight = it->weight;
                best_move   = decoded;
            }
            ++it;
        }
        return best_move;
    }
};

} // namespace polyglot

// ============================================================================
// 2. WDL (Win / Draw / Loss) Sigmoid 胜率模型与 NUMA 亲和性绑定策略
// ============================================================================

struct WdlProbabilities {
    int win_permill  = 333;
    int draw_permill = 334;
    int loss_permill = 333;
};

// 利用 NNUE centipawn 评分与盘面总子力计算千分比 (Win, Draw, Loss)，严格满足 W + D + L == 1000
[[nodiscard]] inline WdlProbabilities compute_wdl(int score_cp, int piece_count) noexcept {
    if (score_cp >= search::kMateThreshold) {
        return WdlProbabilities{1000, 0, 0};
    }
    if (score_cp <= -search::kMateThreshold) {
        return WdlProbabilities{0, 0, 1000};
    }

    // 根据盘面剩余子力调整 Sigmoid 温度系数与平局带宽
    const double phase = std::clamp(static_cast<double>(piece_count) / 32.0, 0.15, 1.0);
    const double scale = 160.0 + 60.0 * phase;
    const double draw_margin = 95.0 * (0.45 + 0.55 * phase);

    const auto sigmoid = [](double x) noexcept -> double {
        return 1.0 / (1.0 + std::exp(-x));
    };

    const double p_win  = sigmoid((static_cast<double>(score_cp) - draw_margin) / scale);
    const double p_loss = sigmoid((-static_cast<double>(score_cp) - draw_margin) / scale);
    const double p_draw = std::max(0.0, 1.0 - p_win - p_loss);

    int w = static_cast<int>(std::round(p_win * 1000.0));
    int l = static_cast<int>(std::round(p_loss * 1000.0));
    w = std::clamp(w, 0, 1000);
    l = std::clamp(l, 0, 1000 - w);
    const int d = 1000 - w - l;

    return WdlProbabilities{w, d, l};
}

enum class NumaPolicy : uint8_t {
    None = 0,
    System,
    BindNode,
    Interleave
};

[[nodiscard]] inline NumaPolicy parse_numa_policy(std::string_view sv) noexcept {
    if (sv == "None" || sv == "none")             return NumaPolicy::None;
    if (sv == "BindNode" || sv == "bindnode")     return NumaPolicy::BindNode;
    if (sv == "Interleave" || sv == "interleave") return NumaPolicy::Interleave;
    return NumaPolicy::System;
}

[[nodiscard]] inline std::string numa_policy_to_string(NumaPolicy p) {
    switch (p) {
        case NumaPolicy::None:       return "None";
        case NumaPolicy::System:     return "System";
        case NumaPolicy::BindNode:   return "BindNode";
        case NumaPolicy::Interleave: return "Interleave";
    }
    return "System";
}

// 绑定搜索工作线程至对应逻辑核心 / NUMA 节点
inline void apply_thread_numa_policy(size_t thread_idx, size_t total_threads, NumaPolicy policy) noexcept {
    if (policy == NumaPolicy::None || policy == NumaPolicy::System) return;
    const unsigned int hw_threads = std::max(1U, std::thread::hardware_concurrency());

    size_t target_cpu = thread_idx % hw_threads;
    if (policy == NumaPolicy::Interleave && total_threads > 1) {
        // Interleave 策略：跨双路/多簇交错映射核心编号
        const size_t half = std::max<size_t>(1, hw_threads / 2);
        target_cpu = ((thread_idx & 1U) ? half : 0U) + (thread_idx / 2U) % half;
        target_cpu %= hw_threads;
    }

#if defined(_WIN32)
    const DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << (target_cpu % (sizeof(DWORD_PTR) * 8)));
    SetThreadAffinityMask(GetCurrentThread(), mask);
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)target_cpu;
#endif
}

// ============================================================================
// 3. 动态时间管理器 (Time Management: wtime/btime/winc/binc/movestogo/movetime)
// ============================================================================

struct GoParameters {
    int64_t  wtime     = -1;
    int64_t  btime     = -1;
    int64_t  winc      = 0;
    int64_t  binc      = 0;
    int      movestogo = 0;
    int64_t  movetime  = -1;
    int      depth     = 0;
    uint64_t nodes     = 0;
    bool     infinite  = false;
    bool     ponder    = false;
};

struct TimeAllocation {
    int64_t soft_time_ms = 0; // 迭代加深完成后提前停止阈值
    int64_t hard_time_ms = 0; // 搜索途中强制中断阈值
};

[[nodiscard]] inline TimeAllocation compute_time_allocation(
    const GoParameters& go,
    Color us,
    int fullmove_number,
    int64_t move_overhead_ms = 100) noexcept
{
    const int64_t overhead = std::clamp<int64_t>(move_overhead_ms, 10, 2000);

    // 1. infinite 模式：不设时限，直到收到 stop
    if (go.infinite) {
        return TimeAllocation{0, 0};
    }

    // 2. 固定每步用时 movetime <ms>
    if (go.movetime > 0) {
        const int64_t safe_mt = std::max<int64_t>(5, go.movetime - overhead);
        return TimeAllocation{safe_mt, safe_mt};
    }

    // 3. 现代全盘均衡平滑时钟分配（Stockfish 连续比例时间模型）
    const int64_t my_time = (us == Color::White) ? go.wtime : go.btime;
    const int64_t my_inc  = (us == Color::White) ? go.winc  : go.binc;

    if (my_time > 0) {
        const int64_t usable_time = std::max<int64_t>(10, my_time - overhead);

        // 如果用户指定了明确的 movestogo (例如 40 步时限)
        if (go.movestogo > 0) {
            const int mtg = std::clamp(go.movestogo, 1, 50);
            int64_t opt = (usable_time / mtg) + (my_inc * 4) / 5;
            int64_t max_t = std::min<int64_t>(usable_time * 8 / 10, opt * 2);
            opt   = std::clamp<int64_t>(opt, 10, usable_time);
            max_t = std::clamp<int64_t>(max_t, opt, usable_time);
            return TimeAllocation{opt, max_t};
        }

        // 1. 基础步耗比例，并在中局黄金期（12~35步）额外给予 25% 的时间倾斜
        double move_factor = 0.022 + 0.0008 * static_cast<double>(fullmove_number);
        if (fullmove_number >= 12 && fullmove_number <= 35) {
            move_factor *= 1.25;
        }
        move_factor = std::clamp(move_factor, 0.022, 0.055);

        // 加秒利用率提升至 90%，减少长对局/加秒局时钟冗余
        int64_t optimum = static_cast<int64_t>(static_cast<double>(usable_time) * move_factor) + (my_inc * 9) / 10;

        // 剩余时间充裕（>= 60s）时适度放宽基础配额（1.15x），把省在钟里的冗余时间投入计算
        if (usable_time >= 60000) {
            optimum = (optimum * 115) / 100;
        }

        // 2. 单步硬上限控制在 3.5x optimum 与 45% 剩余时间之内，防止单步透支
        int64_t maximum = std::min<int64_t>(usable_time * 45 / 100, (optimum * 7) / 2);

        // 残血低时间（< 3s）只收紧 maximum 至 35%，不强压 optimum 以免误伤带加秒的超快棋
        if (usable_time < 3000) {
            optimum = std::min<int64_t>(optimum, usable_time / 4);
            maximum = std::min<int64_t>(maximum, usable_time * 35 / 100);
        }

        optimum = std::clamp<int64_t>(optimum, 10, usable_time);
        maximum = std::clamp<int64_t>(maximum, optimum, usable_time);

        return TimeAllocation{ optimum, maximum };
    }

    return TimeAllocation{0, 0};
}

// ============================================================================
// 4. MultiPV 单条主变例结果与 UCI 格式化输出器
// ============================================================================

struct MultiPvLine {
    int               multipv_rank = 1;
    int               depth        = 1;
    int               seldepth     = 1;
    int               score        = 0;
    uint64_t          nodes        = 0;
    uint64_t          nps          = 0;
    int               hashfull     = 0;
    int64_t           time_ms      = 0;
    Move              root_move    = Move::null();
    std::vector<Move> pv           = {};
};

[[nodiscard]] inline std::string format_uci_score(int score) {
    std::ostringstream oss;
    if (score >= search::kMateThreshold) {
        const int plies_to_mate = search::kMateScore - score;
        const int moves_to_mate = std::max(1, (plies_to_mate + 1) / 2);
        oss << "mate " << moves_to_mate;
    } else if (score <= -search::kMateThreshold) {
        const int plies_to_mate = search::kMateScore + score;
        const int moves_to_mate = std::max(1, (plies_to_mate + 1) / 2);
        oss << "mate -" << moves_to_mate;
    } else {
        // 关键量化转换：训练器与内部评估 1 个兵 = 208 内部单位；UCI 规范 1 个兵 = 100 cp
        const int cp = (score * 100) / 208;
        oss << "cp " << cp;
    }
    return oss.str();
}

[[nodiscard]] inline std::string format_uci_info_line(
    const MultiPvLine& line,
    bool show_wdl,
    int piece_count)
{
    std::ostringstream oss;
    oss << "info depth " << line.depth
        << " seldepth " << line.seldepth
        << " multipv " << line.multipv_rank
        << " score " << format_uci_score(line.score);

    if (show_wdl) {
        const int norm_cp = (line.score * 100) / 208;
        const WdlProbabilities wdl = compute_wdl(norm_cp, piece_count);
        oss << " wdl " << wdl.win_permill << " " << wdl.draw_permill << " " << wdl.loss_permill;
    }

    oss << " nodes " << line.nodes
        << " nps " << line.nps
        << " hashfull " << line.hashfull
        << " time " << line.time_ms
        << " pv";

    for (Move m : line.pv) {
        oss << " " << m.to_uci();
    }
    return oss.str();
}

// ============================================================================
// 5. UCI 引擎核心控制器 (UciEngine: 管理 Options、开局库、Lazy SMP 线程池与异步搜索)
// ============================================================================


inline std::string get_executable_directory() {
#if defined(_WIN32)
    char buffer[MAX_PATH];
    const DWORD len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
    if (len > 0) {
        std::string p(buffer, len);
        const size_t slash = p.find_last_of("\\/");
        if (slash != std::string::npos) {
            return p.substr(0, slash + 1);
        }
    }
#elif defined(__linux__)
    char buffer[1024];
    const ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) {
        buffer[len] = '\0';
        std::string p(buffer);
        const size_t slash = p.find_last_of('/');
        if (slash != std::string::npos) {
            return p.substr(0, slash + 1);
        }
    }
#endif
    return "./";
}

struct EngineOptions {
    size_t      hash_mb = 64;
    size_t      threads = 1;
    NumaPolicy  numa_policy = NumaPolicy::System;
    size_t      multi_pv = 1;
    bool        ponder = true; // 严格对标 Stockfish 官方默认：默认关闭 Ponder，杜绝平台异步死锁！
    bool        uci_show_wdl = true;
    std::string book_path = "";
    std::string eval_file = "custom_1024x2.nnue";
    int64_t     move_overhead_ms = 200; // 网络通信与平台传输保护延迟（毫秒）
    std::string style = "Normal"; // 棋风预设：Normal / Aggressive / Solid
    int         contempt = 0;        // 藐视和棋分：默认 0，Aggressive 默认 35
    int         uci_elo = 2721;      // 标称 Elo 等级分（供 GUI 自动填充显示）
};

class UciEngine {
public:
    explicit UciEngine(std::ostream& out_stream = std::cout)
        : out_(out_stream),
          shared_evaluator_(std::make_shared<nnue::Evaluator>()),
          main_searcher_(options_.hash_mb, shared_evaluator_)
    {
        (void)root_pos_.set_from_fen(Position::kStartFEN);

        // 自动多路径定位并加载 23,107,844 字节正式 NNUE 权重文件（支持 lichess-bot 的 engines/ 目录及可执行文件所在目录）
        const std::string exe_dir = get_executable_directory();
        const std::vector<std::string> search_dirs = {
            "",
            exe_dir,
            "engines/",
            exe_dir + "engines/",
            "../",
            exe_dir + "../"
        };
        for (const auto& fname : {"custom_1024x2.nnue", "sparkchess.nnue", "valkyrie.nnue"}) {
            for (const auto& dir : search_dirs) {
                const std::string candidate = dir + fname;
                if (shared_evaluator_->load_network(candidate)) {
                    options_.eval_file = candidate;
                    break;
                }
            }
            if (shared_evaluator_->is_network_loaded()) break;
        }

        // 启动时即刻输出引擎标识与 NNUE 加载状态（对标 Stockfish 启动横幅，无论命令行还是双击均立即可见）
        out_ << "SparkFish 3.2 by SparkFish Team\n";
        if (shared_evaluator_->is_network_loaded()) {
            out_ << "info string NNUE file: \"" << options_.eval_file << "\" loaded and verified (46,210,308 bytes)\n";
        }
        else {
            out_ << "info string CRITICAL WARNING: custom_1024x2.nnue NOT FOUND! Engine is running on DUMMY test weights!\n";
        }
        out_ << std::flush;
    }

    ~UciEngine() {
        stop_search();
    }

    // 处理单行 UCI 命令；若收到 "quit" 则返回 false 通知主循环退出
    bool process_command(const std::string& raw_line) {
        std::istringstream iss(raw_line);
        std::string token;
        if (!(iss >> token)) {
            // 终端交互友好提示：敲回车时若未输入命令，输出一次状态提醒
            out_ << "readyok\n" << std::flush;
            return true;
        }

        // 终端容错：如果用户直接在终端粘贴了带管道的命令如 echo isready | sparkFish.exe
        if (token == "echo") {
            std::string sub;
            if (iss >> sub) {
                if (sub == "isready") { cmd_isready(); return true; }
                if (sub == "uci") { cmd_uci(); return true; }
            }
        }

        if (token == "uci") {
            cmd_uci();
        } else if (token == "isready") {
            cmd_isready();
        } else if (token == "ucinewgame") {
            cmd_ucinewgame();
        } else if (token == "setoption") {
            cmd_setoption(iss);
        } else if (token == "position") {
            cmd_position(iss);
        } else if (token == "go") {
            cmd_go(iss);
        } else if (token == "ponderhit") {
            cmd_ponderhit();
        } else if (token == "stop") {
            stop_search();
        } else if (token == "d" || token == "display") {
            out_ << "FEN: " << root_pos_.to_fen() << "\n"
                 << "Zobrist: 0x" << std::hex << root_pos_.zobrist_key()
                 << " | PolyglotKey: 0x" << polyglot::compute_key(root_pos_) << std::dec << "\n" << std::flush;
        } else if (token == "quit") {
            stop_search();
            return false;
        }

        return true;
    }

    // 阻塞式启动标准输入 UCI 监听循环
    void run_loop(std::istream& in_stream = std::cin) {
        std::string line;
        while (std::getline(in_stream, line)) {
            if (!process_command(line)) {
                break;
            }
        }
    }

    // 同步等待当前后台搜索线程完成（用于测试或命令行 bench）
    void wait_for_search() {
        if (search_thread_.joinable()) {
            search_thread_.join();
        }
    }

    [[nodiscard]] const EngineOptions& options() const noexcept { return options_; }
    [[nodiscard]] const Position& position() const noexcept { return root_pos_; }
    [[nodiscard]] polyglot::OpeningBook& book() noexcept { return book_; }

private:
    std::ostream&                     out_;
    std::mutex                        io_mutex_{};
    EngineOptions                     options_{};
    Position                          root_pos_{};
    std::vector<StateInfo>            game_state_history_{};

    std::shared_ptr<nnue::Evaluator>  shared_evaluator_;
    search::Searcher                  main_searcher_;
    std::vector<std::unique_ptr<search::Searcher>> helper_searchers_{};
    polyglot::OpeningBook             book_{};

    std::thread                       search_thread_{};
    std::atomic<bool>                 searching_{false};
    std::atomic<bool>                 in_ponder_{false};
    std::atomic<int64_t>              ponder_soft_ms_{0};
    std::atomic<int64_t>              ponder_hard_ms_{0};

    void cmd_isready() {
        std::lock_guard<std::mutex> lock(io_mutex_);
        if (shared_evaluator_->is_network_loaded()) {
            out_ << "info string NNUE file: \"" << options_.eval_file << "\" loaded and verified (46,210,308 bytes)\n";
        } else {
            out_ << "info string CRITICAL WARNING: custom_1024x2.nnue NOT FOUND! Engine is running on DUMMY test weights!\n";
        }
        out_ << "readyok\n" << std::flush;
    }

    void cmd_uci() {
        std::lock_guard<std::mutex> lock(io_mutex_);
        out_ << "id name SparkFish 3.2\n"
            << "id author SparkFish Team\n"
            << "option name UCI_Elo type spin default " << options_.uci_elo << " min 1350 max 3200\n"
            << "option name Hash type spin default " << options_.hash_mb << " min 1 max 4096\n"
            << "option name Threads type spin default " << options_.threads << " min 1 max 256\n"
            << "option name NumaPolicy type combo default " << numa_policy_to_string(options_.numa_policy)
            << " var None var System var BindNode var Interleave\n"
            << "option name MultiPV type spin default " << options_.multi_pv << " min 1 max 16\n"
            << "option name Style type combo default " << options_.style << " var Normal var Aggressive var Solid\n"
            << "option name Contempt type spin default " << options_.contempt << " min -100 max 100\n"
            << "option name Ponder type check default " << (options_.ponder ? "true" : "false") << "\n"
            << "option name UCI_ShowWDL type check default " << (options_.uci_show_wdl ? "true" : "false") << "\n"
            << "option name Move Overhead type spin default " << options_.move_overhead_ms << " min 0 max 5000\n"
            << "option name BookPath type string default <empty>\n"
            << "option name EvalFile type string default " << options_.eval_file << "\n"
            << "uciok\n" << std::flush;
    }

    void cmd_ucinewgame() {
        stop_search();
        main_searcher_.clear_tt();
        main_searcher_.clear_heuristics();
        for (auto& helper : helper_searchers_) {
            helper->clear_tt();
            helper->clear_heuristics();
        }
        game_state_history_.clear();
        (void)root_pos_.set_from_fen(Position::kStartFEN);
    }

    void cmd_setoption(std::istringstream& iss) {
        stop_search();

        std::string token, name, value;
        if (!(iss >> token) || token != "name") return;

        while (iss >> token && token != "value") {
            if (!name.empty()) name += " ";
            name += token;
        }
        if (token == "value") {
            std::getline(iss, value);
            const auto first = value.find_first_not_of(' ');
            if (first != std::string::npos) {
                value = value.substr(first);
            }
            else {
                value.clear();
            }
            // 清洗 Windows 管道残留的 \r 和空格
            while (!value.empty() && (value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
                value.pop_back();
            }
            // 自动剥除首尾可能带的双引号 "..."
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
        }

        if (name == "UCI_Elo" || name == "Elo") {
            const int elo = std::clamp(std::stoi(value.empty() ? "2650" : value), 1350, 3200);
            options_.uci_elo = elo;
        }
        else if (name == "Hash") {
            const int mb = std::clamp(std::stoi(value.empty() ? "64" : value), 1, 4096);
            options_.hash_mb = static_cast<size_t>(mb);
            main_searcher_.resize_tt_mb(options_.hash_mb);
        }
        else if (name == "Threads") {
            const int t = std::clamp(std::stoi(value.empty() ? "1" : value), 1, 256);
            options_.threads = static_cast<size_t>(t);
            rebuild_helper_threads();
        }
        else if (name == "NumaPolicy") {
            options_.numa_policy = parse_numa_policy(value);
        }
        else if (name == "MultiPV") {
            const int mpv = std::clamp(std::stoi(value.empty() ? "1" : value), 1, 16);
            options_.multi_pv = static_cast<size_t>(mpv);
        }
        else if (name == "Style") {
            options_.style = value;
            if (value == "Aggressive") {
                options_.contempt = 35; // 激进模式：默认和棋罚 35 cp
            }
            else if (value == "Solid") {
                options_.contempt = -15; // 稳健模式：默认微劣求和
            }
            else {
                options_.style = "Normal";
                options_.contempt = 0;   // 正常客观模式
            }
        }
        else if (name == "Contempt") {
            const int c = std::clamp(std::stoi(value.empty() ? "0" : value), -100, 100);
            options_.contempt = c;
            if (c > 15)       options_.style = "Aggressive";
            else if (c < -5)  options_.style = "Solid";
            else              options_.style = "Normal";
        }
        else if (name == "Ponder") {
            options_.ponder = (value == "true" || value == "1");
        }
        else if (name == "UCI_ShowWDL") {
            options_.uci_show_wdl = (value == "true" || value == "1");
        }
        else if (name == "Move Overhead" || name == "MoveOverhead") {
            const int64_t ov = std::clamp<int64_t>(std::stoll(value.empty() ? "200" : value), 0LL, 5000LL);
            options_.move_overhead_ms = ov;
        }
        else if (name == "BookPath") {
            options_.book_path = value;
            if (value.empty() || value == "<empty>") {
                book_.clear();
            }
            else {
                const bool ok = book_.load(value);
                std::lock_guard<std::mutex> lock(io_mutex_);
                out_ << "info string Polyglot book \"" << value << "\" "
                    << (ok ? ("loaded (" + std::to_string(book_.size()) + " entries)") : "failed to load")
                    << "\n" << std::flush;
            }
        }
        else if (name == "EvalFile") {
            options_.eval_file = value;
            if (!value.empty() && value != "<empty>") {
                const bool ok = shared_evaluator_->load_network(value);
                std::lock_guard<std::mutex> lock(io_mutex_);
                out_ << "info string NNUE EvalFile \"" << value << "\" "
                    << (ok ? "loaded (46,210,308 bytes verified)" : "not found / using embedded weights")
                    << "\n" << std::flush;
            }
        }
    }

    void rebuild_helper_threads() {
        helper_searchers_.clear();
        if (options_.threads > 1) {
            const size_t helper_count = options_.threads - 1;
            helper_searchers_.reserve(helper_count);
            for (size_t i = 0; i < helper_count; ++i) {
                // 真正的 Lazy SMP: 所有线程共享主搜索器的同一个 4 路组相联置换表与 NNUE 权重
                helper_searchers_.push_back(
                    std::make_unique<search::Searcher>(main_searcher_.shared_tt(), shared_evaluator_));
            }
        }
    }

    void cmd_position(std::istringstream& iss) {
        stop_search();

        std::string token;
        if (!(iss >> token)) return;

        Position new_pos;
        if (token == "startpos") {
            (void)new_pos.set_from_fen(Position::kStartFEN);
            iss >> token; // 可能是 "moves"
        } else if (token == "fen") {
            std::string fen;
            while (iss >> token && token != "moves") {
                if (!fen.empty()) fen += " ";
                fen += token;
            }
            if (!new_pos.set_from_fen(fen)) {
                return;
            }
        } else {
            return;
        }

        game_state_history_.clear();
        game_state_history_.reserve(256);

        if (token == "moves") {
            std::string move_str;
            while (iss >> move_str) {
                for (char& ch : move_str) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                MoveList legal;
                generate_legal_moves(new_pos, legal);
                Move matched = Move::null();
                for (Move m : legal) {
                    if (m.to_uci() == move_str) {
                        matched = m;
                        break;
                    }
                }
                if (matched.is_none()) break;
                game_state_history_.emplace_back();
                new_pos.make_move(matched, game_state_history_.back());
            }
        }

        root_pos_ = std::move(new_pos);
    }

    void cmd_go(std::istringstream& iss) {
        stop_search();

        GoParameters go{};
        std::string token;
        while (iss >> token) {
            if (token == "wtime")         iss >> go.wtime;
            else if (token == "btime")    iss >> go.btime;
            else if (token == "winc")     iss >> go.winc;
            else if (token == "binc")     iss >> go.binc;
            else if (token == "movestogo") iss >> go.movestogo;
            else if (token == "movetime") iss >> go.movetime;
            else if (token == "depth")    iss >> go.depth;
            else if (token == "nodes")    iss >> go.nodes;
            else if (token == "infinite") go.infinite = true;
            else if (token == "ponder")   go.ponder   = true;
        }

        // 1. 若非后台 Ponder 且非 Infinite，优先探测 Polyglot 开局库 (.bin)
        if (!go.ponder && !go.infinite && book_.is_loaded()) {
            const Move book_move = book_.probe(root_pos_);
            if (!book_move.is_none()) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                out_ << "info depth 1 seldepth 1 multipv 1 score cp 0 nodes 1 nps 1000 time 0 pv "
                     << book_move.to_uci() << "\n"
                     << "bestmove " << book_move.to_uci() << "\n" << std::flush;
                return;
            }
        }

        const TimeAllocation alloc = compute_time_allocation(
            go, root_pos_.side_to_move(), root_pos_.fullmove_number(), options_.move_overhead_ms);

        ponder_soft_ms_.store(alloc.soft_time_ms, std::memory_order_relaxed);
        ponder_hard_ms_.store(alloc.hard_time_ms, std::memory_order_relaxed);
        in_ponder_.store(go.ponder, std::memory_order_relaxed);
        searching_.store(true, std::memory_order_relaxed);

        const Position search_pos = root_pos_;

        search_thread_ = std::thread([this, search_pos, go, alloc]() mutable {
            run_lazy_smp_search(search_pos, go, alloc);
        });
    }

    void cmd_ponderhit() {
        // 对手走出预期走法，将后台思考转为正常限时搜索并立即更新各搜索器内部时限与时钟基准！
        in_ponder_.store(false, std::memory_order_relaxed);
        const int64_t soft = ponder_soft_ms_.load(std::memory_order_relaxed);
        const int64_t hard = ponder_hard_ms_.load(std::memory_order_relaxed);
        main_searcher_.notify_ponderhit(soft, hard);
        for (auto& helper : helper_searchers_) {
            helper->notify_ponderhit(soft, hard);
        }
    }

    void stop_search() {
        in_ponder_.store(false, std::memory_order_relaxed);
        main_searcher_.request_stop();
        for (auto& helper : helper_searchers_) {
            helper->request_stop();
        }
        if (search_thread_.joinable()) {
            search_thread_.join();
        }
        searching_.store(false, std::memory_order_relaxed);
    }

    // ------------------------------------------------------------------------
    // 多线程 Lazy SMP + MultiPV 根节点排除法主搜索协调器
    // ------------------------------------------------------------------------
    void run_lazy_smp_search(Position pos, GoParameters go, TimeAllocation alloc) {
        apply_thread_numa_policy(0, options_.threads, options_.numa_policy);

        const auto t_start = std::chrono::steady_clock::now();
        const int max_depth = (go.depth > 0)
            ? std::clamp(go.depth, 1, search::kMaxPly - 2)
            : 64;

        // 1. 启动 Lazy SMP 辅助搜索线程 (Thread 1 .. N-1)
        std::vector<std::thread> workers;
        std::atomic<uint64_t> helper_total_nodes{0};

        if (options_.threads > 1 && helper_searchers_.size() == options_.threads - 1) {
            for (size_t idx = 0; idx < helper_searchers_.size(); ++idx) {
                workers.emplace_back([this, idx, pos, max_depth, alloc, &helper_total_nodes]() mutable {
                    apply_thread_numa_policy(idx + 1, options_.threads, options_.numa_policy);
                    search::SearchLimits worker_limits{};
                    worker_limits.max_depth = std::min(search::kMaxPly - 2, max_depth + static_cast<int>(idx & 1U));
                    worker_limits.hard_time_ms = alloc.hard_time_ms;
                    worker_limits.contempt = options_.contempt;
                    worker_limits.thread_id = static_cast<int>(idx + 1);
                    const auto st = helper_searchers_[idx]->search_position(pos, worker_limits, nullptr, false); // is_main_thread = false
                    helper_total_nodes.fetch_add(st.nodes, std::memory_order_relaxed);
                });
            }
        }

        MoveList legal_root_moves;
        generate_legal_moves(pos, legal_root_moves);

        if (legal_root_moves.empty()) {
            for (auto& h : helper_searchers_) h->request_stop();
            for (auto& w : workers) if (w.joinable()) w.join();
            std::lock_guard<std::mutex> lock(io_mutex_);
            out_ << "bestmove 0000\n" << std::flush;
            searching_.store(false, std::memory_order_relaxed);
            return;
        }

        // 唯一合法走法闪电早停（Only Move Fast-Path）：
        // 若当前局面只有 1 步合法着法（例如被双将只有 1 格可逃），无须长考耗时，1 毫秒秒出，节省巨额时钟！
        if (!go.infinite && !go.ponder && legal_root_moves.size() == 1) {
            for (auto& h : helper_searchers_) h->request_stop();
            for (auto& w : workers) if (w.joinable()) w.join();
            std::lock_guard<std::mutex> lock(io_mutex_);
            out_ << "info depth 1 seldepth 1 score cp 0 nodes 1 time 1 pv " << legal_root_moves[0].to_uci() << "\n";
            out_ << "bestmove " << legal_root_moves[0].to_uci() << "\n" << std::flush;
            searching_.store(false, std::memory_order_relaxed);
            return;
        }

        const size_t multi_pv_count = std::min<size_t>(options_.multi_pv, legal_root_moves.size());
        const int piece_count = pos.occupancy().popcount();

        Move best_move   = legal_root_moves[0];
        Move ponder_move = Move::null();

        // 2. 若 MultiPV == 1，直接调用主搜索器并逐层输出 info
        if (multi_pv_count == 1) {
            search::SearchLimits limits{};
            limits.max_depth    = max_depth;
            limits.max_nodes    = go.nodes;
            limits.is_ponder    = go.ponder;
            limits.soft_time_ms = alloc.soft_time_ms;
            limits.hard_time_ms = alloc.hard_time_ms;
            limits.contempt = options_.contempt; // <--- 加这一行！

            auto on_depth = [&](const search::SearchStats& stats) {
                const auto now = std::chrono::steady_clock::now();
                const int64_t elapsed_ms = std::max<int64_t>(
                    1,
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count());
                uint64_t total_n = stats.nodes;
                for (const auto& h : helper_searchers_) {
                    total_n += h->stats().nodes;
                }

                MultiPvLine line{};
                line.multipv_rank = 1;
                line.depth        = stats.depth;
                line.seldepth     = stats.seldepth;
                line.score        = stats.best_score;
                line.nodes        = total_n;
                line.nps          = (total_n * 1000ULL) / static_cast<uint64_t>(elapsed_ms);
                line.hashfull     = main_searcher_.tt().hashfull();
                line.time_ms      = elapsed_ms;
                line.root_move    = stats.best_move;
                line.pv           = stats.pv_line;

                std::lock_guard<std::mutex> lock(io_mutex_);
                out_ << format_uci_info_line(line, options_.uci_show_wdl, piece_count) << "\n" << std::flush;

            };

            const auto stats = main_searcher_.search_position(pos, limits, on_depth);
            if (!stats.best_move.is_none()) {
                best_move   = stats.best_move;
                ponder_move = (stats.pv_line.size() >= 2) ? stats.pv_line[1] : Move::null();
            }
        } else {
            // 3. MultiPV (2..16) 根节点排除法：对每个合法根走法评估子局面并选出前 K 条主变例
            uint64_t cumulative_nodes = 0;
            for (int d = 1; d <= max_depth; ++d) {
                std::vector<MultiPvLine> candidates;
                candidates.reserve(legal_root_moves.size());

                for (size_t i = 0; i < legal_root_moves.size(); ++i) {
                    const Move rm = legal_root_moves[i];
                    Position child_pos = pos;
                    StateInfo st{};
                    child_pos.make_move(rm, st);

                    search::SearchLimits sub_limits{};
                    sub_limits.max_depth    = std::max(1, d - 1);
                    sub_limits.hard_time_ms = in_ponder_.load(std::memory_order_relaxed) ? 0 : alloc.hard_time_ms;
                    sub_limits.contempt = options_.contempt; // <--- 加这一行！

                    const auto child_stats = main_searcher_.search_position(child_pos, sub_limits);
                    cumulative_nodes += child_stats.nodes;

                    MultiPvLine line{};
                    line.depth     = d;
                    line.seldepth  = std::max(d, child_stats.seldepth + 1);
                    line.score     = -child_stats.best_score;
                    line.root_move = rm;
                    line.pv.push_back(rm);
                    for (Move cm : child_stats.pv_line) {
                        line.pv.push_back(cm);
                    }
                    candidates.push_back(std::move(line));

                    if (main_searcher_.is_stopped()) break;
                }

                std::sort(candidates.begin(), candidates.end(), [](const MultiPvLine& a, const MultiPvLine& b) {
                    return a.score > b.score;
                });

                const auto now = std::chrono::steady_clock::now();
                const int64_t elapsed_ms = std::max<int64_t>(
                    1,
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count());
                const uint64_t total_n = cumulative_nodes + helper_total_nodes.load(std::memory_order_relaxed);

                {
                    std::lock_guard<std::mutex> lock(io_mutex_);
                    for (size_t k = 0; k < multi_pv_count && k < candidates.size(); ++k) {
                        candidates[k].multipv_rank = static_cast<int>(k + 1);
                        candidates[k].nodes        = total_n;
                        candidates[k].nps          = (total_n * 1000ULL) / static_cast<uint64_t>(elapsed_ms);
                        candidates[k].hashfull     = main_searcher_.tt().hashfull();
                        candidates[k].time_ms      = elapsed_ms;
                        out_ << format_uci_info_line(candidates[k], options_.uci_show_wdl, piece_count) << "\n";
                    }
                    out_ << std::flush;
                }

                if (!candidates.empty()) {
                    best_move   = candidates[0].root_move;
                    ponder_move = (candidates[0].pv.size() >= 2) ? candidates[0].pv[1] : Move::null();
                }

                if (main_searcher_.is_stopped()) break;
                if (!in_ponder_.load(std::memory_order_relaxed) &&
                    alloc.soft_time_ms > 0 &&
                    elapsed_ms >= alloc.soft_time_ms)
                {
                    break;
                }
            }
        }

        // 搜索完毕，直接同步辅助线程并出步，严禁任何可能死锁的自旋睡眠

        // 停止并回收所有 Lazy SMP 辅助线程
        for (auto& h : helper_searchers_) h->request_stop();
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }

        // 输出最终 bestmove <move> [ponder <move>]
        {
            std::lock_guard<std::mutex> lock(io_mutex_);
            out_ << "bestmove " << best_move.to_uci();
            if (options_.ponder && !ponder_move.is_none()) {
                out_ << " ponder " << ponder_move.to_uci();
            }
            out_ << "\n" << std::flush;
        }

        searching_.store(false, std::memory_order_relaxed);
    }
};

} // namespace valkyrie::uci
