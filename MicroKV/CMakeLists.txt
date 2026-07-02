cmake_minimum_required(VERSION 3.16)
project(MicroKV LANGUAGES CXX)

add_executable(kv_stored
    src/main.cpp
    src/kv_store.cpp
    src/server.cpp
)

target_include_directories(kv_stored PRIVATE src)
target_compile_features(kv_stored PRIVATE cxx_std_17)
