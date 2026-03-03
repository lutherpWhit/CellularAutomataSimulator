#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <random>
#include <algorithm>

using namespace std;

int applyRule(int left, int center, int right, int rule) {
    /*
    The rule is an 8-bit number where each bit represents the new state of the center cell based on the combination of left, center, and right states.
    The neighborhood is represented as a 3-bit number where:
    */
    int neighborhood = (left << 2) | (center << 1) | right;
    int newState = (rule >> neighborhood) & 1;
    return newState;
    //consider finding a way to have a cell have three states, on, off, and 'turned off' (a cell that was on but is now off, and may be part of a feature). This would allow us to track features more easily, as we could mark cells that have changed state in the current step.
}
struct feature {
    vector<int> states; //List of cells that are part of the feature. Useful for ensures 'lines' of empty cells are correctly identified as features.
    int startIndex; //From left most index of the feature
    int size;
    int state; //state of the feature, may be useful for classification.
    //int max_width; //largest width of the feature at any point in time, may be useful for classification.
    //int length; //number of steps the feature exists for, may be useful for classification.
    int height;
};
struct features{
    vector<feature> Features;
};

map<int, int> categorizeFeatureSizes(const vector<feature>& features) {
    map<int, int> sizeCounts;
    for (const auto& f : features) {
        sizeCounts[f.size]++;   // increment count for this size
    }
    return sizeCounts;
}


void printFeature(const feature& f) {
    cout << "Feature {\n";
    cout << "  startIndex: " << f.startIndex << "\n";
    cout << "  size:       " << f.size << "\n";
    cout << "  state:      " << f.state << "\n";
    cout << "  height:     " << f.height << "\n";

    cout << "  states: [ ";
    for (int s : f.states) cout << s << " ";
    cout << "]\n";

    cout << "}\n";
}

void printFeatures(const vector<feature>& features) {
    cout << "=== Features (" << features.size() << ") ===\n";
    for (size_t i = 0; i < features.size(); ++i) {
        cout << "Feature #" << i << ":\n";
        printFeature(features[i]);
        cout << "\n";
    }
}

void printFeatureSizeCounts(const map<int, int>& sizeCounts) {
    cout << "=== Feature Size Distribution ===\n";
    for (const auto& entry : sizeCounts) {
        cout << "Size " << entry.first 
             << " : " << entry.second << " features\n";
    }
    cout << "=================================\n";
}

int findRightTriangleFeatures(vector<int>& current, vector<int>& previous, int state, vector<feature>& features, int step) {
    /*
    *current: current 1D array of cell states
    *previous: past states of cell states
    *state: which state or color is the feature
    *features: data structure of features and feature characteristic
    *step: what step the feature detection starts at.
    */
    // Need to differentiate between features in white (1) and black (0) cells
    //features are contiguous blocks of same state cells.
    //Takes as input a previous state of the automaton and identifies features in the current state.
    for (size_t i = 0; i < current.size(); ++i) {
        //to create work units for parallel processing we need to insure that we seperate features accros work units so that features are not double counted
        if (current[i] != previous[i] && current[i] == state) { //state represnted the color of the feature.
            feature newFeature;
            newFeature.startIndex = i;
            newFeature.size = 0;
            newFeature.height = step; // Height of the triangle is the number of steps since the feature first appeared
            int length = 0;
            for(int j = i; j < current.size() && current[j] == current[i]; j++) {
                length++;
            }
            int area = (length * (length + 1)) / 2; // Area of the triangle formed by the feature
            newFeature.size = area;
            newFeature.state = current[i];
            features.push_back(newFeature);

            i += length - 1; // Skip the rest of the feature
        }
    }
    return 0;
}

int findEqualTriangleFeatures(vector<int>& current, vector<int>& previous, int state, vector<feature>& features, int step) {
    // Need to differentiate between features in white (1) and black (0) cells
    //features are contiguous blocks of same state cells.
    //Takes as input a previous state of the automaton and identifies features in the current state.
    for (size_t i = 0; i < current.size(); ++i) {
        if (current[i] != previous[i] && current[i] == state) { //state represnted the color of the feature.
            feature newFeature;
            newFeature.startIndex = i;
            newFeature.size = 0;
            newFeature.height = step; // Height of the triangle is the number of steps since the feature first appeared
            int length = 0;
            for(int j = i; j < current.size() && current[j] == current[i]; j++) {
                length++;
            }
            int area = (length * (length + 2)) / 4; // Area of the triangle formed by the feature
            newFeature.size = area;
            newFeature.state = current[i];
            features.push_back(newFeature);
            i += length - 1; // Skip the rest of the feature
        }
    }
    return 0;
}

int imageSimulator(int width, int steps, int rule, vector<int> start, string outputFile = "") {
    const int BLOCK_SIZE = 32;
    //Two arrays may impact how we measure the 'holes' in the rules
    //may need a different approach.
    vector<int> current(width, 0);
    vector<int> next(width, 0);
    vector<feature> features;
    int step = 0;
    bool saveToFile = false;
    bool imageCreated = false;
    bool sizeExceeded = false;
    if ((width > 2000 || steps > 2000) && outputFile != "") {
        cout << "Output size too large, skipping file output." << endl;
        sizeExceeded = true;
    }
    std::ofstream img;
    if (outputFile != "" && !sizeExceeded) {
        if(outputFile.substr(outputFile.find_last_of(".") + 1) == "ppm") {
            img.open(outputFile);
            img << "P3\n";
            img << width << " " << steps + 1 << "\n";
            img << "255\n";
            imageCreated = true;
        } else {
            img.open(outputFile + ".txt");
        }
        saveToFile = true;
    }
    // Initial condition: single active cell in the center
    current = start;
    // Print start row to text file
    if (saveToFile != false) {
        if(imageCreated == false) {
            for (int cell : current) {
                img << (cell ? '#' : ' ');
            }
        img << '\n';
        } else {
            for (int cell : current) {
                img << (cell ? "255 255 255 " : "0 0 0 ");
            }
        img << "\n";
        }
    }
    for (int t = 0; t < steps; ++t) {
        // Compute next state
        #pragma omp parallel for
        for (int i_step = 0; i_step < width; i_step += BLOCK_SIZE) {
            for (int i = i_step; i < i_step + BLOCK_SIZE && i < width; ++i) {
                int left = current[(i - 1 + width) % width];
                int center = current[i];
                int right = current[(i + 1) % width];
                next[i] = applyRule(left, center, right, rule);
            }
        }
        step++;
        if (rule == 110 || rule == 60 || rule == 102 || rule == 124) {
            findRightTriangleFeatures(next, current, 0, features, step);
        }
        if (rule == 90 || rule == 18 || rule == 22 || rule == 26 || rule == 82 || rule == 126) {
            findEqualTriangleFeatures(next, current, 0, features, step);
        }
        current = next;
        //display recently computed cells
        if (saveToFile != false) {
        if(imageCreated == false) {
            for (int cell : current) {
                img << (cell ? '#' : ' ');
            }
            img << '\n';
        } else {
            for (int cell : current) {
                img << (cell ? "255 255 255 " : "0 0 0 ");
            }
            img << "\n";
        }
    }
        
        img << "\n";
    }


    if (saveToFile) {
        img.close();
    }
    //printFeatures(features);
    printFeatureSizeCounts(categorizeFeatureSizes(features));
    return 0;
}

vector<int> randomStart(int width, bool print = false) {

    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist {0, 1};
    
    auto gen = [&](){
                   return dist(mersenne_engine);
               };

    std::vector<int> vec(width);
    std::generate(vec.begin(), vec.end(), gen);
    
    // Optional
    if(print) {
        for (const auto& i : vec) {
            std::cout << i << " ";
        }
    };
    
    return vec;
}



int main() {
    //consider stopping the simulation early if we hit the boundary of the width size
    const int width = 4000;
    const int steps = 2000;

    //for(int rule = 1; rule <= 128; rule++) {
    const int rule = 110;  // Try 90, 110, 184, etc.
    vector<int> start(width, 0);
    //start[width / 2] = 1; //one in the middle
    start[width-1] = 1;  //one on the edge
    //start = randomStart(width);
    //might compare features and feature distribution for center in the middle v.s randomly generated starting conditions.
    //Need options to run the program without visuals.
    auto start_time = std::chrono::high_resolution_clock::now();
    imageSimulator(width, steps, rule, start, "cellular_automaton_output_" + to_string(rule) + ".ppm");
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    
    //}
    return 0;

}