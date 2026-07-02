#include <csignal>
#include <iostream>
#include <string>

#include "server.h"

namespace {

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--socket /tmp/microkv.sock]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);

    std::string socket_path = "/tmp/microkv.sock";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--socket" && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            PrintUsage(argv[0]);
            return 1;
        }
    }

    microkv::Server server(socket_path);
    return server.Run();
}
