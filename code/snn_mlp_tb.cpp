
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

#include "ap_int.h"
#include "fpga_weights.h"

typedef ap_int<16> data_t;
typedef ap_int<32> acc_t;

void snn_mlp(
    volatile data_t *feature_mem,
    volatile ap_uint<8> *prediction,
    volatile acc_t logits_out[NUM_CLASSES]
);

static int label_from_folder(const std::string &name)
{
    if (name == "Full Cable") return 0;
    if (name == "Left End") return 1;
    if (name == "Right End") return 2;
    if (name == "No Cable") return 3;
    return -1;
}

static const char* class_name(int label)
{
    if (label == 0) return "Full Cable";
    if (label == 1) return "Left End";
    if (label == 2) return "Right End";
    if (label == 3) return "No Cable";
    return "Unknown";
}

static bool is_directory(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static bool has_txt_extension(const std::string &name)
{
    if (name.size() < 4) return false;
    return name.substr(name.size() - 4) == ".txt";
}

static bool load_txt_to_bram_array(
    const std::string &txt_path,
    data_t feature_mem[T_STEPS * N_FEATURES]
) {
    std::ifstream file(txt_path.c_str());

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << txt_path << std::endl;
        return false;
    }

    std::string line;
    int count = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (count >= T_STEPS * N_FEATURES) {
            std::cerr << "Too many feature values in " << txt_path << std::endl;
            return false;
        }

        int value = std::stoi(line);
        feature_mem[count] = (data_t)value;
        count++;
    }

    if (count != T_STEPS * N_FEATURES) {
        std::cerr << "Bad feature count in " << txt_path << std::endl;
        std::cerr << "Expected " << T_STEPS * N_FEATURES
                  << " values, got " << count << std::endl;
        return false;
    }

    return true;
}

static std::vector<std::string> list_subfolders(const std::string &root)
{
    std::vector<std::string> folders;

    DIR *dir = opendir(root.c_str());
    if (!dir) {
        std::cerr << "Could not open root folder: " << root << std::endl;
        return folders;
    }

    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;

        if (name == "." || name == "..") continue;

        std::string full_path = root + "/" + name;

        if (is_directory(full_path)) {
            folders.push_back(name);
        }
    }

    closedir(dir);

    std::sort(folders.begin(), folders.end());
    return folders;
}

static std::vector<std::string> list_txt_files(const std::string &folder_path)
{
    std::vector<std::string> files;

    DIR *dir = opendir(folder_path.c_str());
    if (!dir) {
        std::cerr << "Could not open folder: " << folder_path << std::endl;
        return files;
    }

    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;

        if (name == "." || name == "..") continue;
        if (!has_txt_extension(name)) continue;

        std::string full_path = folder_path + "/" + name;
        files.push_back(full_path);
    }

    closedir(dir);

    std::sort(files.begin(), files.end());
    return files;
}

int main()
{
    const int N = 10;

    const std::string root =
        "/home/dinesh/snn_mlp/fpga_test_features";

    int total = 0;
    int correct = 0;

    int confusion[NUM_CLASSES][NUM_CLASSES] = {0};

    std::vector<std::string> folders = list_subfolders(root);

    for (size_t fi = 0; fi < folders.size(); fi++) {
        std::string folder_name = folders[fi];
        std::string folder_path = root + "/" + folder_name;

        int true_label = label_from_folder(folder_name);

        if (true_label < 0) {
            std::cout << "Skipping unknown folder: "
                      << folder_name << std::endl;
            continue;
        }

        std::vector<std::string> txt_files =
            list_txt_files(folder_path);

        int files_to_use =
            std::min((int)txt_files.size(), N);

        std::cout << "\nClass: " << folder_name
                  << " | using " << files_to_use
                  << " files" << std::endl;

        for (int i = 0; i < files_to_use; i++) {
            std::string txt_path = txt_files[i];

            data_t feature_mem[T_STEPS * N_FEATURES];

            ap_uint<8> prediction = 0;
            acc_t logits_out[NUM_CLASSES];

            bool ok = load_txt_to_bram_array(
                txt_path,
                feature_mem
            );

            if (!ok) continue;


            snn_mlp(
                feature_mem,
                &prediction,
                logits_out
            );

            int pred = (int)prediction;

            total++;

            if (pred == true_label) {
                correct++;
            }

            if (pred >= 0 && pred < NUM_CLASSES) {
                confusion[true_label][pred]++;
            }

            std::cout << "[" << total << "] "
                      << txt_path
                      << " | true=" << class_name(true_label)
                      << " | pred=" << class_name(pred)
                      << " | logits=[";

            for (int k = 0; k < NUM_CLASSES; k++) {
                std::cout << (int)logits_out[k];

                if (k != NUM_CLASSES - 1)
                    std::cout << ", ";
            }

            std::cout << "]"
                      << std::endl;

        }
    }

    std::cout << "\n====================================" << std::endl;
    std::cout << "Total tested: " << total << std::endl;
    std::cout << "Correct: " << correct << std::endl;

    if (total > 0) {
        float acc = 100.0f * correct / total;
        std::cout << "Accuracy: " << acc << "%" << std::endl;
    }

    std::cout << "\nConfusion Matrix" << std::endl;
    std::cout << "(rows=true, cols=pred)" << std::endl;

    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            std::cout << confusion[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
