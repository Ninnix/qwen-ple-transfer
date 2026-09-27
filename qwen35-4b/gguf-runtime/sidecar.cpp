#include "llama-batch.h"
#include "llama-qwengram.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::runtime_error("sidecar PLE tokens.bin output.bin chunk_size");
        llama_qwengram_ple ple(argv[1]);
        std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("input open failed");
        size_t bytes = in.tellg();
        if (bytes % sizeof(llama_token)) throw std::runtime_error("invalid token bytes");
        std::vector<llama_token> tokens(bytes / sizeof(llama_token));
        in.seekg(0);
        in.read(reinterpret_cast<char *>(tokens.data()), bytes);
        const size_t chunk = std::stoul(argv[4]);
        if (!chunk) throw std::runtime_error("empty chunk");
        std::ofstream out(argv[3], std::ios::binary);
        if (!out) throw std::runtime_error("output open failed");
        llama_qwengram_state state;
        // Run twice in the same state to exercise position-zero resets.
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (size_t off = 0; off < tokens.size(); off += chunk) {
                size_t n = std::min(chunk, tokens.size() - off);
                std::vector<llama_pos> pos(n);
                std::vector<int32_t> counts(n, 1);
                std::vector<llama_seq_id> seq(n, 0);
                std::vector<llama_seq_id *> ids(n);
                for (size_t i = 0; i < n; ++i) {
                    pos[i] = off + i;
                    ids[i] = &seq[i];
                }
                llama_ubatch batch{};
                batch.n_tokens = n;
                batch.n_pos = 1;
                batch.token = tokens.data() + off;
                batch.pos = pos.data();
                batch.n_seq_id = counts.data();
                batch.seq_id = ids.data();
                std::vector<float> rows(n * 2560);
                ple.fill(batch, state, rows.data());
                out.write(reinterpret_cast<char *>(rows.data()), rows.size() * sizeof(float));
                if (!out) throw std::runtime_error("output write failed");
            }
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
