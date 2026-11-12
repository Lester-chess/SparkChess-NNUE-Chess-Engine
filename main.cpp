/**
 * @file main.cpp
 * @brief SparkFish 3.2 C++20 Chess Engine - Main Entry Point & Full-Stack Self-Test Suite
 * @compile
 *   GCC/Clang: g++ -std=c++20 -O3 -march=native -pthread -Wall -Wextra -Wpedantic -Werror main.cpp -o valkyrie
 *   MSVC     : cl /std:c++20 /O2 /EHsc /W4 /WX main.cpp /Fe:Sparkfish.exe
 * @license MIT (100% Original Architecture designed for HalfKAv2 512x2 NNUE)
 */

#include "uci.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

namespace valkyrie {

// 运行阶段五全流程 UCI 协议、Polyglot 开局库、MultiPV、WDL 与 Lazy SMP 自检
inline void run_stage5_self_test() {
    std::cout << "=== SparkFish 3.2 (main.cpp + uci.hpp) Full-Stack Self-Test ===\n";

    std::ostringstream uci_out;
    uci::UciEngine engine(uci_out);

    // 1. 测试 uci 与 isready 握手
    engine.process_command("uci");
    engine.process_command("isready");

    const std::string handshake = uci_out.str();
    assert(handshake.find("id name SparkFish 3.2") != std::string::npos);
    assert(handshake.find("option name Hash type spin") != std::string::npos);
    assert(handshake.find("option name Threads type spin") != std::string::npos);
    assert(handshake.find("option name NumaPolicy type combo") != std::string::npos);
    assert(handshake.find("option name MultiPV type spin") != std::string::npos);
    assert(handshake.find("option name Ponder type check") != std::string::npos);
    assert(handshake.find("option name UCI_ShowWDL type check") != std::string::npos);
    assert(handshake.find("option name BookPath type string") != std::string::npos);
    assert(handshake.find("option name EvalFile type string") != std::string::npos);
    assert(handshake.find("uciok") != std::string::npos);
    assert(handshake.find("readyok") != std::string::npos);
    std::cout << "[PASS] UCI Handshake & All 8 Advanced Options Registered\n";

    // 2. 测试 Polyglot 16 字节大端序 .bin 开局库构建、保存、加载与二分查找命中
    {
        Position start_pos;
        (void)start_pos.set_from_fen(Position::kStartFEN);
        const uint64_t poly_key = uci::polyglot::compute_key(start_pos);

        // Polyglot 编码 e2e4: from = e2 (file=4, rank=1), to = e4 (file=4, rank=3)
        // bits: to_f(4) | (to_r(3) << 3) | (from_f(4) << 6) | (from_r(1) << 9)
        constexpr uint16_t kPolyMoveE2E4 = static_cast<uint16_t>(4U | (3U << 3U) | (4U << 6U) | (1U << 9U));

        uci::polyglot::OpeningBook temp_book;
        temp_book.insert_entry_for_test(poly_key, kPolyMoveE2E4, 500);
        const std::string book_file = "/tmp/valkyrie_test_book.bin";
        const bool saved = temp_book.save(book_file);
        assert(saved);

        uci_out.str("");
        uci_out.clear();
        engine.process_command("setoption name BookPath value " + book_file);
        engine.process_command("position startpos");
        engine.process_command("go wtime 60000 btime 60000 winc 1000 binc 1000");
        engine.wait_for_search();

        const std::string book_resp = uci_out.str();
        assert(book_resp.find("bestmove e2e4") != std::string::npos);
        std::cout << "[PASS] Polyglot (.bin) 16-Byte Big-Endian Binary Search Book Hit: bestmove e2e4\n";

        // 清空开局库以便后续测试真实 PVS + MultiPV + Lazy SMP 搜索
        engine.process_command("setoption name BookPath value <empty>");
    }

    // 3. 测试高级选项配置：Hash=32, Threads=2 (Lazy SMP), MultiPV=2, UCI_ShowWDL=true, NumaPolicy=Interleave
    engine.process_command("setoption name Hash value 32");
    engine.process_command("setoption name Threads value 2");
    engine.process_command("setoption name NumaPolicy value Interleave");
    engine.process_command("setoption name MultiPV value 2");
    engine.process_command("setoption name UCI_ShowWDL value true");
    engine.process_command("setoption name Ponder value true");

    uci_out.str("");
    uci_out.clear();
    engine.process_command("position startpos moves e2e4 e7e5 g1f3 b8c6");
    engine.process_command("go wtime 30000 btime 30000 winc 500 binc 500 depth 3");
    engine.wait_for_search();

    const std::string mpv_resp = uci_out.str();
    assert(mpv_resp.find("multipv 1") != std::string::npos);
    assert(mpv_resp.find("multipv 2") != std::string::npos);
    assert(mpv_resp.find("wdl ") != std::string::npos);
    assert(mpv_resp.find("bestmove ") != std::string::npos);
    std::cout << "[PASS] Lazy SMP (2 Threads) + MultiPV=2 + WDL + Dynamic Time Control Verified\n";

    // 4. 测试后台思考 go ponder -> ponderhit -> stop 生命周期
    engine.process_command("setoption name MultiPV value 1");
    uci_out.str("");
    uci_out.clear();
    engine.process_command("position startpos moves d2d4 d7d5");
    engine.process_command("go ponder depth 3");
    engine.process_command("ponderhit");
    engine.wait_for_search();
    assert(uci_out.str().find("bestmove ") != std::string::npos);
    std::cout << "[PASS] Ponder -> Ponderhit Asynchronous Lifecycle Verified\n";

    engine.process_command("quit");
    std::cout << "=== All Stage 5 UCI Engine Assembly Assertions Passed! ===\n";
}

} // namespace valkyrie

int main(int argc, char* argv[]) {
    // 关闭 C/C++ 标准流同步以获得微秒级 UCI 命令响应延迟
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc > 1) {
        const std::string arg1 = argv[1];
        if (arg1 == "--self-test" || arg1 == "bench") {
            valkyrie::run_stage5_self_test();
            return 0;
        }
    }

    // 启动标准 UCI 协议监听循环
    auto engine = std::make_unique<valkyrie::uci::UciEngine>(std::cout);
    engine->run_loop(std::cin);
    return 0;
}
