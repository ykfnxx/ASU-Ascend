#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "kv_store.h"

namespace microkv {

class Server {
public:
    explicit Server(std::string socket_path);
    ~Server();

    int Run();

private:
    bool HandleClient(int client_fd);
    bool HandleRequest(int client_fd, const std::uint8_t* header);
    bool SendResponse(int client_fd, Status status, const std::vector<std::uint8_t>& payload);
    bool ReadKey(int client_fd, Key* key);
    bool ReadBytes(int client_fd, std::uint32_t len, std::vector<std::uint8_t>* out);

    std::string socket_path_;
    int server_fd_ = -1;
    std::mutex store_mutex_;
    KVStore store_;
};

}  // namespace microkv
