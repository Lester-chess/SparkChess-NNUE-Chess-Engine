/**
 * @file position.hpp
 * @brief Valkyrie C++20 Chess Engine - Stage 1: Core Types, Bitboard, 16-bit Move,
 *        45,056-Dim HalfKAv2 NNUE Indexing, StateInfo & Make/Undo Position Core.
 * @standard C++20 (-std=c++20 / /std:c++20)
 * @license MIT (100% Original Architecture designed for HalfKAv2 512x2 NNUE)
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_MSC_VER)
#include <cstdlib> // for _byteswap_uint64
#endif

namespace valkyrie {

// ============================================================================
// 0. C++20 64位字节序翻转兼容封装 (替代 C++23 std::byteswap)
// ============================================================================

[[nodiscard]] constexpr uint64_t bswap64(uint64_t x) noexcept {
    if (std::is_constant_evaluated()) {
        return ((x & 0x00000000000000FFULL) << 56U) |
               ((x & 0x000000000000FF00ULL) << 40U) |
               ((x & 0x0000000000FF0000ULL) << 24U) |
               ((x & 0x00000000FF000000ULL) <<  8U) |
               ((x & 0x000000FF00000000ULL) >>  8U) |
               ((x & 0x0000FF0000000000ULL) >> 24U) |
               ((x & 0x00FF000000000000ULL) >> 40U) |
               ((x & 0xFF00000000000000ULL) >> 56U);
    } else {
#if defined(_MSC_VER)
        return _byteswap_uint64(x);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_bswap64(x);
#else
        return ((x & 0x00000000000000FFULL) << 56U) |
               ((x & 0x000000000000FF00ULL) << 40U) |
               ((x & 0x0000000000FF0000ULL) << 24U) |
               ((x & 0x00000000FF000000ULL) <<  8U) |
               ((x & 0x000000FF00000000ULL) >>  8U) |
               ((x & 0x0000FF0000000000ULL) >> 24U) |
               ((x & 0x00FF000000000000ULL) >> 40U) |
               ((x & 0xFF00000000000000ULL) >> 56U);
#endif
    }
}

// ============================================================================
// 1. 基础枚举与核心常量定义 (Color, PieceType, Square, CastlingRights)
// ============================================================================

enum class Color : uint8_t {
    White = 0,
    Black = 1,
    None  = 2
};

[[nodiscard]] constexpr Color operator~(Color c) noexcept {
    return (c == Color::White) ? Color::Black : (c == Color::Black ? Color::White : Color::None);
}

enum class PieceType : uint8_t {
    Pawn   = 0,
    Knight = 1,
    Bishop = 2,
    Rook   = 3,
    Queen  = 4,
    King   = 5,
    None   = 6
};

// 12种具体棋子编码 (0..5: 白子 P,N,B,R,Q,K; 6..11: 黑子 p,n,b,r,q,k; 12: Empty)
enum class Piece : uint8_t {
    WhitePawn = 0, WhiteKnight = 1, WhiteBishop = 2, WhiteRook = 3, WhiteQueen = 4, WhiteKing = 5,
    BlackPawn = 6, BlackKnight = 7, BlackBishop = 8, BlackRook = 9, BlackQueen = 10, BlackKing = 11,
    None = 12
};

[[nodiscard]] constexpr Piece make_piece(Color c, PieceType pt) noexcept {
    if (c == Color::None || pt == PieceType::None) return Piece::None;
    return static_cast<Piece>(static_cast<uint8_t>(c) * 6U + static_cast<uint8_t>(pt));
}

[[nodiscard]] constexpr Color piece_color(Piece p) noexcept {
    if (p == Piece::None) return Color::None;
    return (static_cast<uint8_t>(p) < 6U) ? Color::White : Color::Black;
}

[[nodiscard]] constexpr PieceType piece_type(Piece p) noexcept {
    if (p == Piece::None) return PieceType::None;
    return static_cast<PieceType>(static_cast<uint8_t>(p) % 6U);
}

[[nodiscard]] constexpr char piece_to_char(Piece p) noexcept {
    constexpr std::string_view kChars = "PNBRQKpnbrqk.";
    const auto idx = static_cast<size_t>(p);
    return (idx < kChars.size()) ? kChars[idx] : '.';
}

[[nodiscard]] constexpr Piece char_to_piece(char ch) noexcept {
    constexpr std::string_view kChars = "PNBRQKpnbrqk";
    const auto pos = kChars.find(ch);
    return (pos != std::string_view::npos) ? static_cast<Piece>(pos) : Piece::None;
}

// Square: Little-Endian Rank-File Mapping (LERF: a1=0, h1=7, a8=56, h8=63)
enum class Square : uint8_t {
    a1 = 0,  b1, c1, d1, e1, f1, g1, h1,
    a2 = 8,  b2, c2, d2, e2, f2, g2, h2,
    a3 = 16, b3, c3, d3, e3, f3, g3, h3,
    a4 = 24, b4, c4, d4, e4, f4, g4, h4,
    a5 = 32, b5, c5, d5, e5, f5, g5, h5,
    a6 = 40, b6, c6, d6, e6, f6, g6, h6,
    a7 = 48, b7, c7, d7, e7, f7, g7, h7,
    a8 = 56, b8, c8, d8, e8, f8, g8, h8,
    None = 64
};

[[nodiscard]] constexpr uint8_t sq_index(Square s) noexcept {
    return static_cast<uint8_t>(s);
}

[[nodiscard]] constexpr uint8_t file_of(Square s) noexcept {
    return static_cast<uint8_t>(s) & 7U;
}

[[nodiscard]] constexpr uint8_t rank_of(Square s) noexcept {
    return static_cast<uint8_t>(s) >> 3U;
}

[[nodiscard]] constexpr Square make_square(uint8_t file, uint8_t rank) noexcept {
    return static_cast<Square>((rank << 3U) | (file & 7U));
}

// 垂直翻转 (sq ^ 56)：用于黑白视角对称转换及 HalfKAv2 NNUE 特征视角归一化
[[nodiscard]] constexpr Square flip_vertical(Square s) noexcept {
    return (s == Square::None) ? Square::None : static_cast<Square>(static_cast<uint8_t>(s) ^ 56U);
}

[[nodiscard]] inline std::string square_to_string(Square s) {
    if (s == Square::None || sq_index(s) >= 64U) return "-";
    std::string out(2, ' ');
    out[0] = static_cast<char>('a' + file_of(s));
    out[1] = static_cast<char>('1' + rank_of(s));
    return out;
}

[[nodiscard]] constexpr Square string_to_square(std::string_view sv) noexcept {
    if (sv.size() < 2) return Square::None;
    const char f = sv[0];
    const char r = sv[1];
    if (f < 'a' || f > 'h' || r < '1' || r > '8') return Square::None;
    return make_square(static_cast<uint8_t>(f - 'a'), static_cast<uint8_t>(r - '1'));
}

// 棋盘方向偏移枚举
enum class Direction : int8_t {
    North      =  8,
    South      = -8,
    East       =  1,
    West       = -1,
    NorthEast  =  9,
    NorthWest  =  7,
    SouthEast  = -7,
    SouthWest  = -9,
    NorthNorth = 16,
    SouthSouth = -16
};

// 易位权限位掩码 (4-bit)
enum class CastlingRights : uint8_t {
    None            = 0,
    WhiteKingside   = 1 << 0, // K (0b0001)
    WhiteQueenside  = 1 << 1, // Q (0b0010)
    BlackKingside   = 1 << 2, // k (0b0100)
    BlackQueenside  = 1 << 3, // q (0b1000)
    WhiteBoth       = WhiteKingside | WhiteQueenside,
    BlackBoth       = BlackKingside | BlackQueenside,
    All             = WhiteBoth | BlackBoth
};

[[nodiscard]] constexpr CastlingRights operator|(CastlingRights a, CastlingRights b) noexcept {
    return static_cast<CastlingRights>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
[[nodiscard]] constexpr CastlingRights operator&(CastlingRights a, CastlingRights b) noexcept {
    return static_cast<CastlingRights>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}
[[nodiscard]] constexpr CastlingRights operator~(CastlingRights a) noexcept {
    return static_cast<CastlingRights>(~static_cast<uint8_t>(a) & 0x0FU);
}
constexpr CastlingRights& operator|=(CastlingRights& a, CastlingRights b) noexcept {
    a = a | b;
    return a;
}
constexpr CastlingRights& operator&=(CastlingRights& a, CastlingRights b) noexcept {
    a = a & b;
    return a;
}

// ============================================================================
// 2. Bitboard 位棋盘封装类 (uint64_t 零开销封装 + C++20 <bit> / 跨平台 bswap64)
// ============================================================================

class Bitboard {
public:
    static constexpr uint64_t kFileA = 0x0101010101010101ULL;
    static constexpr uint64_t kFileB = kFileA << 1U;
    static constexpr uint64_t kFileG = kFileA << 6U;
    static constexpr uint64_t kFileH = kFileA << 7U;
    static constexpr uint64_t kNotFileA  = ~kFileA;
    static constexpr uint64_t kNotFileH  = ~kFileH;
    static constexpr uint64_t kNotFileAB = ~(kFileA | kFileB);
    static constexpr uint64_t kNotFileGH = ~(kFileG | kFileH);

    static constexpr uint64_t kRank1 = 0x00000000000000FFULL;
    static constexpr uint64_t kRank4 = kRank1 << (8U * 3U);
    static constexpr uint64_t kRank5 = kRank1 << (8U * 4U);
    static constexpr uint64_t kRank8 = kRank1 << (8U * 7U);

    constexpr Bitboard() noexcept : bits_(0ULL) {}
    constexpr explicit Bitboard(uint64_t raw) noexcept : bits_(raw) {}

    [[nodiscard]] static constexpr Bitboard from_square(Square s) noexcept {
        return (sq_index(s) < 64U) ? Bitboard(1ULL << sq_index(s)) : Bitboard(0ULL);
    }

    [[nodiscard]] constexpr uint64_t raw() const noexcept { return bits_; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return bits_ != 0ULL; }
    [[nodiscard]] constexpr bool empty() const noexcept { return bits_ == 0ULL; }
    [[nodiscard]] constexpr bool any() const noexcept { return bits_ != 0ULL; }

    // 位操作
    constexpr void set(Square s) noexcept {
        assert(sq_index(s) < 64U);
        bits_ |= (1ULL << sq_index(s));
    }

    constexpr void clear(Square s) noexcept {
        assert(sq_index(s) < 64U);
        bits_ &= ~(1ULL << sq_index(s));
    }

    constexpr void toggle(Square s) noexcept {
        assert(sq_index(s) < 64U);
        bits_ ^= (1ULL << sq_index(s));
    }

    [[nodiscard]] constexpr bool test(Square s) const noexcept {
        assert(sq_index(s) < 64U);
        return (bits_ & (1ULL << sq_index(s))) != 0ULL;
    }

    // C++20 <bit> 内联硬件指令：popcount / countr_zero / countl_zero
    [[nodiscard]] constexpr int popcount() const noexcept {
        return std::popcount(bits_);
    }

    [[nodiscard]] constexpr Square lsb() const noexcept {
        assert(bits_ != 0ULL);
        return static_cast<Square>(std::countr_zero(bits_));
    }

    [[nodiscard]] constexpr Square msb() const noexcept {
        assert(bits_ != 0ULL);
        return static_cast<Square>(63 - std::countl_zero(bits_));
    }

    constexpr Square pop_lsb() noexcept {
        assert(bits_ != 0ULL);
        const auto s = lsb();
        bits_ &= (bits_ - 1ULL);
        return s;
    }

    // 垂直翻转整个位棋盘 (a1<->a8)：使用兼容 GCC/Clang (__builtin_bswap64) 与 MSVC (_byteswap_uint64) 的 bswap64
    [[nodiscard]] constexpr Bitboard flip_vertical() const noexcept {
        return Bitboard(bswap64(bits_));
    }

    // 编译期方向安全移位（自动屏蔽 A列/H列 越界回绕）
    template <Direction Dir>
    [[nodiscard]] constexpr Bitboard shift() const noexcept {
        if constexpr (Dir == Direction::North)           return Bitboard(bits_ << 8U);
        else if constexpr (Dir == Direction::South)      return Bitboard(bits_ >> 8U);
        else if constexpr (Dir == Direction::NorthNorth) return Bitboard(bits_ << 16U);
        else if constexpr (Dir == Direction::SouthSouth) return Bitboard(bits_ >> 16U);
        else if constexpr (Dir == Direction::East)       return Bitboard((bits_ & kNotFileH) << 1U);
        else if constexpr (Dir == Direction::West)       return Bitboard((bits_ & kNotFileA) >> 1U);
        else if constexpr (Dir == Direction::NorthEast)  return Bitboard((bits_ & kNotFileH) << 9U);
        else if constexpr (Dir == Direction::NorthWest)  return Bitboard((bits_ & kNotFileA) << 7U);
        else if constexpr (Dir == Direction::SouthEast)  return Bitboard((bits_ & kNotFileH) >> 7U);
        else if constexpr (Dir == Direction::SouthWest)  return Bitboard((bits_ & kNotFileA) >> 9U);
        else return Bitboard(0ULL);
    }

    // 运行时方向安全移位
    [[nodiscard]] constexpr Bitboard shifted(Direction dir) const noexcept {
        switch (dir) {
            case Direction::North:      return shift<Direction::North>();
            case Direction::South:      return shift<Direction::South>();
            case Direction::East:       return shift<Direction::East>();
            case Direction::West:       return shift<Direction::West>();
            case Direction::NorthEast:  return shift<Direction::NorthEast>();
            case Direction::NorthWest:  return shift<Direction::NorthWest>();
            case Direction::SouthEast:  return shift<Direction::SouthEast>();
            case Direction::SouthWest:  return shift<Direction::SouthWest>();
            case Direction::NorthNorth: return shift<Direction::NorthNorth>();
            case Direction::SouthSouth: return shift<Direction::SouthSouth>();
        }
        return Bitboard(0ULL);
    }

    // 范围 for 循环迭代器支持：for (Square sq : bb) { ... }
    class Iterator {
    public:
        constexpr explicit Iterator(uint64_t b) noexcept : bits_(b) {}
        [[nodiscard]] constexpr Square operator*() const noexcept {
            return static_cast<Square>(std::countr_zero(bits_));
        }
        constexpr Iterator& operator++() noexcept {
            bits_ &= (bits_ - 1ULL);
            return *this;
        }
        [[nodiscard]] constexpr bool operator!=(const Iterator& other) const noexcept {
            return bits_ != other.bits_;
        }
    private:
        uint64_t bits_;
    };

    [[nodiscard]] constexpr Iterator begin() const noexcept { return Iterator(bits_); }
    [[nodiscard]] constexpr Iterator end() const noexcept { return Iterator(0ULL); }

    // 运算符重载
    [[nodiscard]] constexpr Bitboard operator|(Bitboard rhs) const noexcept { return Bitboard(bits_ | rhs.bits_); }
    [[nodiscard]] constexpr Bitboard operator&(Bitboard rhs) const noexcept { return Bitboard(bits_ & rhs.bits_); }
    [[nodiscard]] constexpr Bitboard operator^(Bitboard rhs) const noexcept { return Bitboard(bits_ ^ rhs.bits_); }
    [[nodiscard]] constexpr Bitboard operator~() const noexcept { return Bitboard(~bits_); }
    constexpr Bitboard& operator|=(Bitboard rhs) noexcept { bits_ |= rhs.bits_; return *this; }
    constexpr Bitboard& operator&=(Bitboard rhs) noexcept { bits_ &= rhs.bits_; return *this; }
    constexpr Bitboard& operator^=(Bitboard rhs) noexcept { bits_ ^= rhs.bits_; return *this; }

    // 严格遵循要求：参数类型为 const Bitboard& rhs
    [[nodiscard]] constexpr bool operator==(const Bitboard& rhs) const noexcept {
        return bits_ == rhs.bits_;
    }
    [[nodiscard]] constexpr bool operator!=(const Bitboard& rhs) const noexcept {
        return bits_ != rhs.bits_;
    }

private:
    uint64_t bits_;
};

// ============================================================================
// 3. Move 结构体 (16-bit 紧凑打包：from_sq:6, to_sq:6, flags:4)
// ============================================================================
// Bit Layout:
//   bits [ 0.. 5] : from_sq (0..63)
//   bits [ 6..11] : to_sq   (0..63)
//   bits [12..15] : MoveFlag (4 bits, 涵盖 Quiet, DoublePawnPush, Castling, EnPassant, Promotions)

enum class MoveFlag : uint8_t {
    Quiet              = 0b0000, // 0: 普通走子
    DoublePawnPush     = 0b0001, // 1: 兵双步推进
    KingCastle         = 0b0010, // 2: 王翼易位 (O-O)
    QueenCastle        = 0b0011, // 3: 后翼易位 (O-O-O)
    Capture            = 0b0100, // 4: 普通吃子
    EnPassantCapture   = 0b0101, // 5: 吃过路兵
    PromoKnight        = 0b1000, // 8: 升变为马
    PromoBishop        = 0b1001, // 9: 升变为象
    PromoRook          = 0b1010, // 10: 升变为车
    PromoQueen         = 0b1011, // 11: 升变为后
    PromoCaptureKnight = 0b1100, // 12: 吃子并升变为马
    PromoCaptureBishop = 0b1101, // 13: 吃子并升变为象
    PromoCaptureRook   = 0b1110, // 14: 吃子并升变为车
    PromoCaptureQueen  = 0b1111  // 15: 吃子并升变为后
};

struct Move {
    uint16_t data_ = 0;

    constexpr Move() noexcept = default;
    constexpr explicit Move(uint16_t raw) noexcept : data_(raw) {}

    constexpr Move(Square from, Square to, MoveFlag flags = MoveFlag::Quiet) noexcept
        : data_(static_cast<uint16_t>(
              (static_cast<uint16_t>(from) & 0x3FU) |
              ((static_cast<uint16_t>(to) & 0x3FU) << 6U) |
              ((static_cast<uint16_t>(flags) & 0x0FU) << 12U))) {}

    [[nodiscard]] static constexpr Move null() noexcept { return Move(0); }

    [[nodiscard]] constexpr bool is_none() const noexcept { return data_ == 0; }
    [[nodiscard]] constexpr uint16_t raw() const noexcept { return data_; }

    [[nodiscard]] constexpr Square from_sq() const noexcept {
        return static_cast<Square>(data_ & 0x3FU);
    }

    [[nodiscard]] constexpr Square to_sq() const noexcept {
        return static_cast<Square>((data_ >> 6U) & 0x3FU);
    }

    [[nodiscard]] constexpr MoveFlag flags() const noexcept {
        return static_cast<MoveFlag>((data_ >> 12U) & 0x0FU);
    }

    [[nodiscard]] constexpr bool is_capture() const noexcept {
        return (static_cast<uint8_t>(flags()) & 0b0100U) != 0U;
    }

    [[nodiscard]] constexpr bool is_promotion() const noexcept {
        return (static_cast<uint8_t>(flags()) & 0b1000U) != 0U;
    }

    [[nodiscard]] constexpr bool is_en_passant() const noexcept {
        return flags() == MoveFlag::EnPassantCapture;
    }

    [[nodiscard]] constexpr bool is_castling() const noexcept {
        const auto f = flags();
        return f == MoveFlag::KingCastle || f == MoveFlag::QueenCastle;
    }

    [[nodiscard]] constexpr bool is_double_pawn_push() const noexcept {
        return flags() == MoveFlag::DoublePawnPush;
    }

    [[nodiscard]] constexpr PieceType promotion_type() const noexcept {
        if (!is_promotion()) return PieceType::None;
        // 低 2 位 0..3 对应 Knight(1), Bishop(2), Rook(3), Queen(4)
        return static_cast<PieceType>(((static_cast<uint8_t>(flags()) & 0b0011U) + 1U));
    }

    [[nodiscard]] inline std::string to_uci() const {
        if (is_none()) return "0000";
        std::string s = square_to_string(from_sq()) + square_to_string(to_sq());
        if (is_promotion()) {
            constexpr char kPromoChars[] = {' ', 'n', 'b', 'r', 'q'};
            s.push_back(kPromoChars[static_cast<uint8_t>(promotion_type())]);
        }
        return s;
    }

    [[nodiscard]] constexpr bool operator==(const Move& rhs) const noexcept {
        return data_ == rhs.data_;
    }
    [[nodiscard]] constexpr bool operator!=(const Move& rhs) const noexcept {
        return data_ != rhs.data_;
    }
};

// ============================================================================
// 4. Zobrist 局面哈希预置随机数表 (编译期 SplitMix64 生成，零运行时初始化开销)
// ============================================================================

namespace zobrist {

struct Table {
    std::array<std::array<uint64_t, 64>, 12> piece_sq{};
    std::array<uint64_t, 16> castling{};
    std::array<uint64_t, 8>  ep_file{};
    uint64_t side_to_move{};
};

[[nodiscard]] consteval uint64_t splitmix64_next(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

[[nodiscard]] consteval Table generate_table() noexcept {
    Table t{};
    uint64_t seed = 0x56414C4B59524945ULL; // "VALKYRIE" ASCII hex seed
    for (size_t p = 0; p < 12; ++p) {
        for (size_t sq = 0; sq < 64; ++sq) {
            t.piece_sq[p][sq] = splitmix64_next(seed);
        }
    }
    for (size_t c = 0; c < 16; ++c) {
        t.castling[c] = splitmix64_next(seed);
    }
    for (size_t f = 0; f < 8; ++f) {
        t.ep_file[f] = splitmix64_next(seed);
    }
    t.side_to_move = splitmix64_next(seed);
    return t;
}

inline constexpr Table kKeys = generate_table();

} // namespace zobrist

// ============================================================================
// 5. HalfKAv2 512x2 NNUE 特征索引计算器 (严格对标 45,056 维，64王桶 x 11平面 x 64格)
// ============================================================================
// 维度规范:
//   - 64 个王格桶 (无水平镜像 / No horizontal flip)
//   - 每个王桶 11 个棋子平面 (11 * 64 = 704 维)，总维度 = 64 * 704 = 45,056
//   - 平面 0~9 (非王棋子): p_idx = static_cast<int>(type) * 2 + (color != perspective)
//   - 平面 10 (王平面):    己方王与敌方王均映射到 p_idx = 10
//   - 索引公式: sq + p_idx * 64 + ksq * 704 (黑方视角下 sq 和 ksq 垂直翻转 ^ 56)

namespace nnue {

inline constexpr int kNumKingBuckets = 64;
inline constexpr int kNumPlanes      = 11;
inline constexpr int kBucketStride   = kNumPlanes * 64;          // 704
inline constexpr int kHalfKAv2Dims   = kNumKingBuckets * kBucketStride; // 45,056

struct FeaturePair {
    Square sq;
    Piece  piece;
    int    white_plane;
    int    black_plane;
    int    white_feature_idx;
    int    black_feature_idx;
};

[[nodiscard]] constexpr int halfkav2_plane(Color perspective, Piece p) noexcept {
    const PieceType type  = piece_type(p);
    const Color     color = piece_color(p);
    if (type == PieceType::King) {
        return 10; // 己方王和敌方王均映射到平面 10
    }
    return static_cast<int>(type) * 2 + (color != perspective ? 1 : 0);
}

[[nodiscard]] constexpr int halfkav2_index(Color perspective, Square king_sq, Piece p, Square sq) noexcept {
    assert(p != Piece::None && king_sq != Square::None && sq != Square::None);

    const int norm_ksq = (perspective == Color::Black)
        ? static_cast<int>(sq_index(flip_vertical(king_sq)))
        : static_cast<int>(sq_index(king_sq));

    const int norm_sq = (perspective == Color::Black)
        ? static_cast<int>(sq_index(flip_vertical(sq)))
        : static_cast<int>(sq_index(sq));

    const int p_idx = halfkav2_plane(perspective, p);
    const int idx   = norm_sq + p_idx * 64 + norm_ksq * 704;

    assert(idx >= 0 && idx < kHalfKAv2Dims);
    return idx;
}

} // namespace nnue

// ============================================================================
// 6. StateInfo 不可逆状态保存结构体 (支持零拷贝 O(1) make_move / undo_move)
// ============================================================================

struct StateInfo {
    CastlingRights castling       = CastlingRights::None;
    Square         ep_square      = Square::None;
    uint16_t       halfmove_clock = 0;
    Piece          captured_piece = Piece::None;
    uint64_t       zobrist_key    = 0ULL;
};

// ============================================================================
// 7. Position 局面类 (12种棋子位棋盘 + 双色占位 + FEN + Make/Undo + Zobrist)
// ============================================================================

class Position {
public:
    static constexpr std::string_view kStartFEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    Position() noexcept {
        clear();
    }

    // 重置为空棋盘（注意：空棋盘默认带 castling_[0] 的 Zobrist 基准值，使得任意时刻 zobrist_key_ == compute_zobrist_from_scratch()）
    void clear() noexcept {
        piece_bbs_.fill(Bitboard(0ULL));
        color_bbs_.fill(Bitboard(0ULL));
        mailbox_.fill(Piece::None);
        side_to_move_ = Color::White;
        castling_     = CastlingRights::None;
        ep_square_    = Square::None;
        halfmove_clk_ = 0;
        fullmove_num_ = 1;
        zobrist_key_  = zobrist::kKeys.castling[0];
        history_keys_.clear();
    }

    // 解析标准 FEN 字符串，成功返回 true，失败返回 false 并保持原局面不变
    [[nodiscard]] bool set_from_fen(std::string_view fen_str) {
        Position temp;
        // 将 zobrist_key_ 置零，解析完棋子、走子方、易位权、过路兵后一次性精确异或，杜绝重复异或 kKeys.castling[0]
        temp.zobrist_key_ = 0ULL;

        std::string fen_copy(fen_str);
        std::istringstream iss(fen_copy);

        std::string board_part, side_part, castle_part, ep_part;
        if (!(iss >> board_part >> side_part >> castle_part >> ep_part)) {
            return false;
        }

        // 1. 解析棋盘布局 (从 a8 到 h1)
        int rank = 7;
        int file = 0;
        for (char ch : board_part) {
            if (ch == '/') {
                if (file != 8 || rank <= 0) return false;
                --rank;
                file = 0;
            } else if (ch >= '1' && ch <= '8') {
                file += (ch - '0');
                if (file > 8) return false;
            } else {
                const Piece p = char_to_piece(ch);
                if (p == Piece::None || file >= 8 || rank < 0) return false;
                temp.put_piece(p, make_square(static_cast<uint8_t>(file), static_cast<uint8_t>(rank)));
                ++file;
            }
        }
        if (rank != 0 || file != 8) return false;

        // 2. 解析走子方
        if (side_part == "w") {
            temp.side_to_move_ = Color::White;
        } else if (side_part == "b") {
            temp.side_to_move_ = Color::Black;
            temp.zobrist_key_ ^= zobrist::kKeys.side_to_move;
        } else {
            return false;
        }

        // 3. 解析王车易位权（仅异或一次最终解析出的 castling_ 掩码）
        temp.castling_ = CastlingRights::None;
        if (castle_part != "-") {
            for (char ch : castle_part) {
                switch (ch) {
                    case 'K': temp.castling_ |= CastlingRights::WhiteKingside;  break;
                    case 'Q': temp.castling_ |= CastlingRights::WhiteQueenside; break;
                    case 'k': temp.castling_ |= CastlingRights::BlackKingside;  break;
                    case 'q': temp.castling_ |= CastlingRights::BlackQueenside; break;
                    default: return false;
                }
            }
        }
        temp.zobrist_key_ ^= zobrist::kKeys.castling[static_cast<uint8_t>(temp.castling_)];

        // 4. 解析过路兵目标格
        if (ep_part == "-") {
            temp.ep_square_ = Square::None;
        } else {
            const Square ep = string_to_square(ep_part);
            if (ep == Square::None || (rank_of(ep) != 2U && rank_of(ep) != 5U)) return false;
            temp.ep_square_ = ep;
            temp.zobrist_key_ ^= zobrist::kKeys.ep_file[file_of(ep)];
        }

        // 5. 解析半回合计数与全回合数（可选字段，缺失时默认 0 和 1）
        int hm = 0, fm = 1;
        if (iss >> hm) {
            temp.halfmove_clk_ = static_cast<uint16_t>(std::max(0, hm));
            if (iss >> fm) {
                temp.fullmove_num_ = static_cast<uint16_t>(std::max(1, fm));
            }
        }

        assert(temp.zobrist_key_ == temp.compute_zobrist_from_scratch());
        *this = std::move(temp);
        return true;
    }

    // 导出标准 FEN 字符串
    [[nodiscard]] std::string to_fen() const {
        std::ostringstream oss;
        for (int rank = 7; rank >= 0; --rank) {
            int empty_cnt = 0;
            for (int file = 0; file < 8; ++file) {
                const Piece p = piece_on(make_square(static_cast<uint8_t>(file), static_cast<uint8_t>(rank)));
                if (p == Piece::None) {
                    ++empty_cnt;
                } else {
                    if (empty_cnt > 0) {
                        oss << empty_cnt;
                        empty_cnt = 0;
                    }
                    oss << piece_to_char(p);
                }
            }
            if (empty_cnt > 0) oss << empty_cnt;
            if (rank > 0) oss << '/';
        }

        oss << ' ' << (side_to_move_ == Color::White ? 'w' : 'b') << ' ';

        if (castling_ == CastlingRights::None) {
            oss << '-';
        } else {
            if ((castling_ & CastlingRights::WhiteKingside)  != CastlingRights::None) oss << 'K';
            if ((castling_ & CastlingRights::WhiteQueenside) != CastlingRights::None) oss << 'Q';
            if ((castling_ & CastlingRights::BlackKingside)  != CastlingRights::None) oss << 'k';
            if ((castling_ & CastlingRights::BlackQueenside) != CastlingRights::None) oss << 'q';
        }

        oss << ' ' << square_to_string(ep_square_)
            << ' ' << halfmove_clk_
            << ' ' << fullmove_num_;
        return oss.str();
    }

    // 在指定格放置棋子并增量更新位棋盘与 Zobrist Key
    constexpr void put_piece(Piece p, Square s) noexcept {
        assert(p != Piece::None && sq_index(s) < 64U);
        const auto p_idx = static_cast<size_t>(p);
        const auto c_idx = static_cast<size_t>(piece_color(p));
        piece_bbs_[p_idx].set(s);
        color_bbs_[c_idx].set(s);
        mailbox_[sq_index(s)] = p;
        zobrist_key_ ^= zobrist::kKeys.piece_sq[p_idx][sq_index(s)];
    }

    // 移除指定格上的棋子并增量更新位棋盘与 Zobrist Key
    constexpr void remove_piece(Square s) noexcept {
        assert(sq_index(s) < 64U);
        const Piece p = mailbox_[sq_index(s)];
        if (p == Piece::None) return;
        const auto p_idx = static_cast<size_t>(p);
        const auto c_idx = static_cast<size_t>(piece_color(p));
        piece_bbs_[p_idx].clear(s);
        color_bbs_[c_idx].clear(s);
        mailbox_[sq_index(s)] = Piece::None;
        zobrist_key_ ^= zobrist::kKeys.piece_sq[p_idx][sq_index(s)];
    }

    // 高效执行走子 make_move(Move, StateInfo&)：保存不可逆状态至 st 并增量更新位棋盘与 Zobrist
    
    // 原生空步走子与撤销 (支持零开销 Null Move Pruning，杜绝 dummy_move == 0 伪空步崩溃)
    void make_null_move(StateInfo& st) noexcept {
        st.castling       = castling_;
        st.ep_square      = ep_square_;
        st.halfmove_clock = halfmove_clk_;
        st.captured_piece = Piece::None;
        st.zobrist_key    = zobrist_key_;
        history_keys_.push_back(zobrist_key_);

        if (ep_square_ != Square::None) {
            zobrist_key_ ^= zobrist::kKeys.ep_file[file_of(ep_square_)];
            ep_square_ = Square::None;
        }

        ++halfmove_clk_;
        side_to_move_ = ~side_to_move_;
        zobrist_key_ ^= zobrist::kKeys.side_to_move;
    }

    void undo_null_move(const StateInfo& st) noexcept {
        side_to_move_ = ~side_to_move_;
        castling_     = st.castling;
        ep_square_    = st.ep_square;
        halfmove_clk_ = st.halfmove_clock;
        zobrist_key_  = st.zobrist_key;
        if (!history_keys_.empty()) {
            history_keys_.pop_back();
        }
    }

    void make_move(Move m, StateInfo& st) {
        assert(!m.is_none());

        // 1. 保存当前不可逆状态到 StateInfo
        st.castling       = castling_;
        st.ep_square      = ep_square_;
        st.halfmove_clock = halfmove_clk_;
        st.captured_piece = Piece::None;
        st.zobrist_key    = zobrist_key_;

        history_keys_.push_back(zobrist_key_);

        const Square   from  = m.from_sq();
        const Square   to    = m.to_sq();
        const MoveFlag fl    = m.flags();
        const Piece    moved = piece_on(from);
        const Color    us    = side_to_move_;
        const Color    them  = ~us;

        assert(moved != Piece::None && piece_color(moved) == us);

        // 2. 清除旧的过路兵 Zobrist 贡献
        if (ep_square_ != Square::None) {
            zobrist_key_ ^= zobrist::kKeys.ep_file[file_of(ep_square_)];
            ep_square_ = Square::None;
        }

        // 3. 更新 50 回合计数与全回合数
        if (piece_type(moved) == PieceType::Pawn || m.is_capture()) {
            halfmove_clk_ = 0;
        } else {
            ++halfmove_clk_;
        }
        if (us == Color::Black) {
            ++fullmove_num_;
        }

        // 4. 处理吃子（含过路兵吃子与普通/升变吃子）
        if (fl == MoveFlag::EnPassantCapture) {
            const Square cap_sq = (us == Color::White)
                ? static_cast<Square>(sq_index(to) - 8U)
                : static_cast<Square>(sq_index(to) + 8U);
            st.captured_piece = piece_on(cap_sq);
            assert(st.captured_piece == make_piece(them, PieceType::Pawn));
            remove_piece(cap_sq);
        } else if (m.is_capture()) {
            st.captured_piece = piece_on(to);
            assert(st.captured_piece != Piece::None && piece_color(st.captured_piece) == them);
            remove_piece(to);
        }

        // 5. 移动己方棋子（含升变处理）
        remove_piece(from);
        if (m.is_promotion()) {
            put_piece(make_piece(us, m.promotion_type()), to);
        } else {
            put_piece(moved, to);
        }

        // 6. 处理王车易位时的车移动
        if (fl == MoveFlag::KingCastle) {
            const Square r_from = (us == Color::White) ? Square::h1 : Square::h8;
            const Square r_to   = (us == Color::White) ? Square::f1 : Square::f8;
            remove_piece(r_from);
            put_piece(make_piece(us, PieceType::Rook), r_to);
        } else if (fl == MoveFlag::QueenCastle) {
            const Square r_from = (us == Color::White) ? Square::a1 : Square::a8;
            const Square r_to   = (us == Color::White) ? Square::d1 : Square::d8;
            remove_piece(r_from);
            put_piece(make_piece(us, PieceType::Rook), r_to);
        }

        // 7. 处理双步兵推进产生的过路兵目标格
        if (fl == MoveFlag::DoublePawnPush) {
            ep_square_ = (us == Color::White)
                ? static_cast<Square>(sq_index(from) + 8U)
                : static_cast<Square>(sq_index(from) - 8U);
            zobrist_key_ ^= zobrist::kKeys.ep_file[file_of(ep_square_)];
        }

        // 8. 更新易位权并增量更新 Zobrist
        zobrist_key_ ^= zobrist::kKeys.castling[static_cast<uint8_t>(castling_)];
        auto revoke_by_sq = [&](Square s) noexcept {
            if (s == Square::e1)      castling_ &= ~CastlingRights::WhiteBoth;
            else if (s == Square::a1) castling_ &= ~CastlingRights::WhiteQueenside;
            else if (s == Square::h1) castling_ &= ~CastlingRights::WhiteKingside;
            else if (s == Square::e8) castling_ &= ~CastlingRights::BlackBoth;
            else if (s == Square::a8) castling_ &= ~CastlingRights::BlackQueenside;
            else if (s == Square::h8) castling_ &= ~CastlingRights::BlackKingside;
        };
        revoke_by_sq(from);
        revoke_by_sq(to);
        zobrist_key_ ^= zobrist::kKeys.castling[static_cast<uint8_t>(castling_)];

        // 9. 切换走子方
        side_to_move_ = them;
        zobrist_key_ ^= zobrist::kKeys.side_to_move;

        assert(zobrist_key_ == compute_zobrist_from_scratch());
    }

    // 高效悔棋 undo_move(Move, const StateInfo&)：精确回滚棋盘与不可逆状态，为 Perft 与 Alpha-Beta 搜索提供支撑
    void undo_move(Move m, const StateInfo& st) noexcept {
        assert(!m.is_none());

        // 1. 恢复走子方（当前 side_to_move_ 是对方，回退为走子方 us）
        const Color us = ~side_to_move_;
        side_to_move_  = us;

        if (us == Color::Black) {
            --fullmove_num_;
        }

        const Square   from = m.from_sq();
        const Square   to   = m.to_sq();
        const MoveFlag fl   = m.flags();

        // 2. 若是王车易位，先把车退回原位
        if (fl == MoveFlag::KingCastle) {
            const Square r_from = (us == Color::White) ? Square::h1 : Square::h8;
            const Square r_to   = (us == Color::White) ? Square::f1 : Square::f8;
            remove_piece(r_to);
            put_piece(make_piece(us, PieceType::Rook), r_from);
        } else if (fl == MoveFlag::QueenCastle) {
            const Square r_from = (us == Color::White) ? Square::a1 : Square::a8;
            const Square r_to   = (us == Color::White) ? Square::d1 : Square::d8;
            remove_piece(r_to);
            put_piece(make_piece(us, PieceType::Rook), r_from);
        }

        // 3. 将到达格 to 上的棋子移回起点格 from（若是升变则退回为兵）
        const Piece moved_on_to = piece_on(to);
        remove_piece(to);
        if (m.is_promotion()) {
            put_piece(make_piece(us, PieceType::Pawn), from);
        } else {
            put_piece(moved_on_to, from);
        }

        // 4. 恢复被吃掉的棋子（含过路兵吃子与常规吃子）
        if (fl == MoveFlag::EnPassantCapture) {
            const Square cap_sq = (us == Color::White)
                ? static_cast<Square>(sq_index(to) - 8U)
                : static_cast<Square>(sq_index(to) + 8U);
            assert(st.captured_piece != Piece::None);
            put_piece(st.captured_piece, cap_sq);
        } else if (st.captured_piece != Piece::None) {
            put_piece(st.captured_piece, to);
        }

        // 5. 从 StateInfo 恢复易位权、过路兵格、50回合计数与精确 Zobrist 哈希
        castling_     = st.castling;
        ep_square_    = st.ep_square;
        halfmove_clk_ = st.halfmove_clock;
        zobrist_key_  = st.zobrist_key;

        if (!history_keys_.empty()) {
            history_keys_.pop_back();
        }

        assert(zobrist_key_ == compute_zobrist_from_scratch());
    }

    // 从头完整重算 Zobrist 哈希（用于调试与一致性断言校验）
    [[nodiscard]] uint64_t compute_zobrist_from_scratch() const noexcept {
        uint64_t h = 0ULL;
        for (size_t p = 0; p < 12; ++p) {
            for (Square s : piece_bbs_[p]) {
                h ^= zobrist::kKeys.piece_sq[p][sq_index(s)];
            }
        }
        h ^= zobrist::kKeys.castling[static_cast<uint8_t>(castling_)];
        if (ep_square_ != Square::None) {
            h ^= zobrist::kKeys.ep_file[file_of(ep_square_)];
        }
        if (side_to_move_ == Color::Black) {
            h ^= zobrist::kKeys.side_to_move;
        }
        return h;
    }

    // 精确 FIDE 重复判定：区分搜索树内部循环与真实对局历史
    // 1. 若在搜索树内 (i <= ply) 遇到相同局面，2 次即构成循环（和棋）
    // 2. 若在对局历史中 (i > ply)，必须此前已出现至少 2 次（本步为第 3 次），才构成三次重复和棋！
    [[nodiscard]] bool is_repetition_draw(int ply) const noexcept {
        const int end = std::min<int>(halfmove_clk_, static_cast<int>(history_keys_.size()));
        int past_game_count = 0;
        for (int i = 2; i <= end; i += 2) {
            if (history_keys_[history_keys_.size() - static_cast<size_t>(i)] == zobrist_key_) {
                if (i <= ply) {
                    return true; // 搜索树内局部循环走法直接判和
                }
                ++past_game_count;
                if (past_game_count >= 2) {
                    return true; // 对局历史中已出现 2 次，加上当前为第 3 次，构成正式 3-fold repetition
                }
            }
        }
        return false;
    }

    [[nodiscard]] int repetition_count() const noexcept {
        int count = 1;
        for (auto it = history_keys_.rbegin(); it != history_keys_.rend(); ++it) {
            if (*it == zobrist_key_) ++count;
        }
        return count;
    }

    // 提取当前局面的 HalfKAv2 512x2 NNUE 激活特征列表 (严格对标 45,056 维)
    [[nodiscard]] std::vector<nnue::FeaturePair> extract_halfkav2_features() const {
        std::vector<nnue::FeaturePair> features;
        features.reserve(32);
        const Bitboard w_king_bb = pieces(Color::White, PieceType::King);
        const Bitboard b_king_bb = pieces(Color::Black, PieceType::King);
        if (w_king_bb.empty() || b_king_bb.empty()) return features;

        const Square w_ksq = w_king_bb.lsb();
        const Square b_ksq = b_king_bb.lsb();

        for (Square s : occupancy()) {
            const Piece p = piece_on(s);
            features.push_back(nnue::FeaturePair{
                s,
                p,
                nnue::halfkav2_plane(Color::White, p),
                nnue::halfkav2_plane(Color::Black, p),
                nnue::halfkav2_index(Color::White, w_ksq, p, s),
                nnue::halfkav2_index(Color::Black, b_ksq, p, s)
            });
        }
        return features;
    }

    // 位棋盘与状态只读访问器
    [[nodiscard]] constexpr Bitboard pieces(Piece p) const noexcept {
        return piece_bbs_[static_cast<size_t>(p)];
    }
    [[nodiscard]] constexpr Bitboard pieces(Color c, PieceType pt) const noexcept {
        return piece_bbs_[static_cast<size_t>(make_piece(c, pt))];
    }
    [[nodiscard]] constexpr Bitboard pieces(Color c) const noexcept {
        return color_bbs_[static_cast<size_t>(c)];
    }
    [[nodiscard]] constexpr Bitboard occupancy() const noexcept {
        return color_bbs_[0] | color_bbs_[1];
    }
    [[nodiscard]] constexpr Piece piece_on(Square s) const noexcept {
        return (sq_index(s) < 64U) ? mailbox_[sq_index(s)] : Piece::None;
    }
    [[nodiscard]] constexpr Color side_to_move() const noexcept { return side_to_move_; }
    [[nodiscard]] constexpr CastlingRights castling_rights() const noexcept { return castling_; }
    [[nodiscard]] constexpr Square ep_square() const noexcept { return ep_square_; }
    [[nodiscard]] constexpr uint16_t halfmove_clock() const noexcept { return halfmove_clk_; }
    [[nodiscard]] constexpr uint16_t fullmove_number() const noexcept { return fullmove_num_; }
    [[nodiscard]] constexpr uint64_t zobrist_key() const noexcept { return zobrist_key_; }

private:
    std::array<Bitboard, 12> piece_bbs_{};
    std::array<Bitboard, 2>  color_bbs_{};
    std::array<Piece, 64>    mailbox_{};

    Color          side_to_move_ = Color::White;
    CastlingRights castling_     = CastlingRights::None;
    Square         ep_square_    = Square::None;
    uint16_t       halfmove_clk_ = 0;
    uint16_t       fullmove_num_ = 1;
    uint64_t       zobrist_key_  = 0ULL;

    std::vector<uint64_t> history_keys_{};
};

} // namespace valkyrie

// ============================================================================
// 8. 阶段一自检与单元测试入口 (当定义 VALKYRIE_STAGE1_MAIN 时编译为独立可执行文件)
//    g++ -std=c++20 -O3 -Wall -Wextra -Wpedantic -Werror -x c++ -DVALKYRIE_STAGE1_MAIN position.hpp -o valkyrie_test
// ============================================================================
#ifdef VALKYRIE_STAGE1_MAIN
int main() {
    using namespace valkyrie;

    Position pos;
    const bool ok = pos.set_from_fen(Position::kStartFEN);
    assert(ok);
    assert(pos.to_fen() == Position::kStartFEN);
    assert(pos.zobrist_key() == pos.compute_zobrist_from_scratch());

    const uint64_t init_hash = pos.zobrist_key();

    std::cout << "=== Valkyrie C++20 Engine (position.hpp) - Stage 1 Self-Test ===\n";
    std::cout << "Initial FEN     : " << pos.to_fen() << "\n";
    std::cout << "Zobrist Key     : 0x" << std::hex << std::setw(16) << std::setfill('0')
              << init_hash << " ( scratch match: YES )" << std::dec << "\n";
    std::cout << "HalfKAv2 Dims   : " << nnue::kHalfKAv2Dims << " (64 buckets x 11 planes x 64 sq)\n";

    // 测试 make_move 与 undo_move 完整可逆性：1. e2e4 e7e5 2. g1f3
    StateInfo st1{}, st2{}, st3{};
    const Move m1(Square::e2, Square::e4, MoveFlag::DoublePawnPush);
    const Move m2(Square::e7, Square::e5, MoveFlag::DoublePawnPush);
    const Move m3(Square::g1, Square::f3, MoveFlag::Quiet);

    pos.make_move(m1, st1);
    assert(pos.zobrist_key() == pos.compute_zobrist_from_scratch());
    pos.make_move(m2, st2);
    assert(pos.zobrist_key() == pos.compute_zobrist_from_scratch());
    pos.make_move(m3, st3);
    assert(pos.zobrist_key() == pos.compute_zobrist_from_scratch());

    std::cout << "After 3 plies   : " << pos.to_fen() << "\n";

    pos.undo_move(m3, st3);
    pos.undo_move(m2, st2);
    pos.undo_move(m1, st1);

    assert(pos.to_fen() == Position::kStartFEN);
    assert(pos.zobrist_key() == init_hash);
    std::cout << "After 3 undos   : " << pos.to_fen() << " (Hash restored: 0x"
              << std::hex << pos.zobrist_key() << std::dec << ")\n";

    return 0;
}
#endif
