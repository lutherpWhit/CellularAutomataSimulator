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

void featuresTallyInit(map<int, int>& features_tally, int n) {
    for(int i = 0; i<= n; i++) {
        int T = (i * (i + 1)) / 2; //max feature size for a feature of length i
        features_tally[T] = 0; //init all triangular numbers up to n as keys
        //used so that we can use atomic operations to update the map.
    }
}

void printFeatureSizeCounts(const map<int, int>& sizeCounts, int totalCellAverage, std::string outputFile = "") {
    //Prints the distribution of feature sizes to the console and optionally to a text file.
    std::ofstream img;
    if (outputFile == "") {
        outputFile = "feature_size_distribution.txt";
        //default output file name if none provided, may want to change this to include a timestamp or rule number to avoid overwriting previous outputs.
    }
    if (!outputFile.empty()) {
        img.open(outputFile);
    }

    cout << "=== Feature Size Distribution ===\n";
    for (const auto& entry : sizeCounts) {
        if (entry.second > 0) {
            cout << "Size " << entry.first 
                 << " : " << entry.second << " features\n";
        }
    }
    cout << "=================================\n";

    if (img.is_open()) {
        for (const auto& entry : sizeCounts) {
            img << "Size " << entry.first << ": " << entry.second << " features\n";
        }
        img << "Average percentage of state 0 cells per row: " << totalCellAverage << "%\n";
        img.close();
    }
}

void printFeatureSizeCountsList(const vector<int>& sizeCounts, int totalCellAverage, std::string outputFile = "") {
    //Prints the distribution of feature sizes to the console and optionally to a text file.
    std::ofstream img;
    if (outputFile == "") {
        outputFile = "feature_size_distribution.txt";
        //default output file name if none provided, may want to change this to include a timestamp or rule number to avoid overwriting previous outputs.
    }
    if (!outputFile.empty()) {
        img.open(outputFile);
    }

    cout << "=== Feature Size Distribution ===\n";
    for (int i=0; i < sizeCounts.size(); i++) {
        if (sizeCounts[i] > 0) {
            cout << "Size " << i 
                 << " : " << sizeCounts[i] << " features\n";
        }
    }
    cout << "=================================\n";

    if (img.is_open()) {
        for (int i=0; i < sizeCounts.size(); i++) {
            if (sizeCounts[i] > 0) {
                img << "Size " << i << ": " << sizeCounts[i] << " features\n";
            }
        }
    }
    img << "Average percentage of state 0 cells per row: " << totalCellAverage << "%\n";
    img.close();
}


void saveAverageTally(const map<int, int>& average_tally, std::string filename = "") {
    //prints the distribution of average percentage of state 0 cells per row to a csv file, with two columns, average percentage of state 0 cells and count.
    std::ofstream file;
    if (filename == "") {
        filename = "average_tally.csv"; //default output file name if none provided, may want to change this to include a timestamp or rule number to avoid overwriting previous outputs.
    }
    file.open(filename);
    if (file.is_open()) {
        file << "Average Percentage of State 0 Cells,Count\n";
        for (const auto& entry : average_tally) {
            file << entry.first << "," << entry.second << "\n";
        }
        file.close();
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
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
    //Saves features into the feature data structure, more memory intensive.
    
    int BLOCK_SIZE = 100;
    #pragma omp parallel 
    {
        vector<feature> local_features; // Each thread will have its own local vector to store features
        #pragma omp for schedule(static)
        for(int i_step = 0; i_step < current.size(); i_step += BLOCK_SIZE) {
            int blockEnd = min(i_step + BLOCK_SIZE, (int)current.size());
            //consider parallelizing this loop, but need to be careful about features that may span across work units.
            int i = i_step;
            if(i_step > 0 && current[i_step-1] == state) {
                while(i < blockEnd && current[i] == state) {
                    i++; //skip over any initial cells in the block that are the same state, to avoid starting a feature in the middle of a block of same state cells.
                }
            }
            for(; i < blockEnd; i++) {
                if (current[i] != previous[i] && current[i] == state) { //state represnted the color of the feature.
                    feature newFeature;
                    newFeature.startIndex = i;
                    newFeature.size = 0;
                    newFeature.height = step; // Height of the triangle is the number of steps since the feature first appeared
                    int length = 0;
                    for(int j = i; j < current.size() && current[j] == current[i]; j++) {
                        length++; //continue to count the length of the feature until we hit a different state cell.
                    }
                    int area = (length * (length + 1)) / 2; // Area of the triangle formed by the feature
                    newFeature.size = area;
                    newFeature.state = current[i];
                    local_features.push_back(newFeature);

                    i += length - 1; // Skip the rest of the feature
                }
            }
        }
        #pragma omp critical 
        features.insert(features.end(), local_features.begin(), local_features.end());
    
    }
    return 0;
}

int findRightTriangleFeaturesTally(vector<int>& current, vector<int>& previous, int state, map<int, int>& features_tally, int step) {
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
    //Saves data as a count of feature sizes rather than a list of features, to save memory and allow for larger simulations.
    
    int BLOCK_SIZE = 100; //increase
    #pragma omp parallel 
    {
        map<int, int> local_tally;
        #pragma omp for schedule(static) //dynamic?
        for(int i_step = 0; i_step < current.size(); i_step += BLOCK_SIZE) {
            int blockEnd = min(i_step + BLOCK_SIZE, (int)current.size());
            //consider parallelizing this loop, but need to be careful about features that may span across work units.
            int i = i_step;
            if(i_step > 0 && current[i_step-1] == state) {
                while(i < blockEnd && current[i] == state) {
                    i++; //skip over any initial cells in the block that are the same state, to avoid starting a feature in the middle of a block of same state cells.
                }
            }
            for(; i < blockEnd; i++) {
                if (current[i] != previous[i] && current[i] == state) { //state represnted the color of the feature.
                    int length = 0;
                    for(int j = i; j < current.size() && current[j] == current[i]; j++) {
                        length++; //continue to count the length of the feature until we hit a different state cell.
                    }
                    int area = (length * (length + 1)) / 2; // Area of the triangle formed by the feature
                    local_tally[area]++;

                    i += length - 1; // Skip the rest of the feature
                }
            }
        }
        //replace with atomic operations on a concurrent hash map or something similar to reduce contention and improve performance.
        //look into a map reduce function (openmp) mapReduce.
        
        {
            for(auto &p : local_tally) {
                //check return value of brackets [] operator
                auto update = features_tally[p.first]; //reference
                #pragma omp atomic
                update += p.second;
                //try atomic
                features_tally[p.first] = update;
            }
        }
        
        
    
    }
    return 0;
}

int findRightTriangleFeaturesTally_Op2(vector<int>& current, vector<int>& previous, int state, vector<int>& features_tally, int step) {
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
    //Saves data as a count of feature sizes rather than a list of features, to save memory and allow for larger simulations.
    
    int BLOCK_SIZE = 100; //increase 500?
    #pragma omp parallel 
    {
        //map<int, int> local_tally;
        #pragma omp for schedule(static) //dynamic?
        for(int i_step = 0; i_step < current.size(); i_step += BLOCK_SIZE) {
            int blockEnd = min(i_step + BLOCK_SIZE, (int)current.size());
            //consider parallelizing this loop, but need to be careful about features that may span across work units.
            int i = i_step;
            if(i_step > 0 && current[i_step-1] == state) {
                while(i < blockEnd && current[i] == state) {
                    i++; //skip over any initial cells in the block that are the same state, to avoid starting a feature in the middle of a block of same state cells.
                }
            }
            for(; i < blockEnd; i++) {
                if (current[i] != previous[i] && current[i] == state) { //state represnted the color of the feature.
                    int length = 0;
                    for(int j = i; j < current.size() && current[j] == current[i]; j++) {
                        length++; //continue to count the length of the feature until we hit a different state cell.
                    }
                    int area = (length * (length + 1)) / 2; // Area of the triangle formed by the feature
                    //local_tally[area]++;
                    #pragma omp atomic update
                    features_tally[area]++;

                    i += length - 1; // Skip the rest of the feature
                }
            }
        }
        //replace with atomic operations on a concurrent hash map or something similar to reduce contention and improve performance.
        //look into a map reduce function (openmp) mapReduce.
        /*
        {
            for(auto &p : local_tally) {
                //check return value of brackets [] operator
                auto update = features_tally[p.first];
                #pragma omp atomic
                update += p.second;
                features_tally[p.first] = update;
            }
        }
        */
        
        
    
    }
    return 0;
}

int findRightTriangleFeaturesTally_UnOp(vector<int>& current, vector<int>& previous, int state, map<int, int>& features_tally, int step) {
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
    //Saves data as a count of feature sizes rather than a list of features, to save memory and allow for larger simulations.
    
    int BLOCK_SIZE = 100; //increase
    #pragma omp parallel 
    {
        map<int, int> local_tally;
        #pragma omp for schedule(static) //dynamic?
        for(int i_step = 0; i_step < current.size(); i_step += BLOCK_SIZE) {
            int blockEnd = min(i_step + BLOCK_SIZE, (int)current.size());
            //consider parallelizing this loop, but need to be careful about features that may span across work units.
            int i = i_step;
            if(i_step > 0 && current[i_step-1] == state) {
                while(i < blockEnd && current[i] == state) {
                    i++; //skip over any initial cells in the block that are the same state, to avoid starting a feature in the middle of a block of same state cells.
                }
            }
            for(; i < blockEnd; i++) {
                if (current[i] != previous[i] && current[i] == state) { //state represnted the color of the feature.
                    int length = 0;
                    for(int j = i; j < current.size() && current[j] == current[i]; j++) {
                        length++; //continue to count the length of the feature until we hit a different state cell.
                    }
                    int area = (length * (length + 1)) / 2; // Area of the triangle formed by the feature
                    local_tally[area]++;

                    i += length - 1; // Skip the rest of the feature
                }
            }
        }
        #pragma omp critical //Slowdown
        //replace with atomic operations on a concurrent hash map or something similar to reduce contention and improve performance.
        //look into a map reduce function (openmp) mapReduce.
        
        {
            for(auto &p : local_tally) {
                features_tally[p.first] += p.second;
            }
        }
    
    }
    return 0;
}

int findRightTriangleFeatures_nonParallel(vector<int>& current, vector<int>& previous, int state, vector<feature>& features, int step) {
    //Used to debug parallel version of findRightTriangleFeatures, should produce the same output.
    
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

int rowAverage(vector<int>& current, vector<int>& previous, int state, int width) {
    //Iterates through the row and determines the percentage of white and black cells
    int BLOCK_SIZE = 100;
    int numStateCells = 0;
    #pragma omp for
    //Check if this is causing errors in parallelizing
    for (int i_step = 0; i_step < current.size(); i_step += BLOCK_SIZE) {
        for (int i = i_step; i < i_step + BLOCK_SIZE && i < current.size(); i++) {
            /*
            if (current[i] != previous[i] && current[i] == state) {
                activeCells[i] = 1; // Mark active cells that have changsed to the specified state
            }
            */
            if (current[i] == state) {
                numStateCells++;
            }
        }
    }
    //check to see if there is no parallelization errors by comparing the output of this function with a non-parallelized version.
    return (numStateCells * 100) / width;
}

map<int, int> imageSimulator(int width, int steps, int rule, vector<int> start, map<int, int>& average_tally,
                            string outputFile = "", string logFile = "", int op = 0) {
    const int BLOCK_SIZE = 32;
    //Two arrays may impact how we measure the 'holes' in the rules
    //may need a different approach.
    vector<int> current(width, 0);
    vector<int> next(width, 0);
    //vector<int> activeCells(width, 0); // Track cells that have changed to the specified state, may be useful for feature classification.
    vector<feature> features;
    vector<feature> features2;
    map<int, int> features_tally;
    vector<int> features_tally_list(width/2, 0); //index represents feature size, value at index represents count of features of that size. May be faster than using a map with atomic operations, but uses more memory.
    
    int n = 50;
    //featuresTallyInit(features_tally, n); // Initialize the features tally with triangular numbers up to n.

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
    if (saveToFile) {
        if(imageCreated == false) {
            //Saves output as a text file with '#' representing state 1 cells and ' ' representing state 0 cells (largely unused)
            for (int cell : current) {
                img << (cell ? '#' : ' ');
            }
        img << '\n';
        } else {
            //Saves output as a ppm image file, with white pixels representing state 1 cells and black pixels representing state 0 cells.
            for (int cell : current) {
                img << (cell ? "255 255 255 " : "0 0 0 ");
            }
        img << "\n";
        }
    }
    int totalCellAverage = 0;
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
            if (op == 0) {
                findRightTriangleFeatures_nonParallel(next, current, 0, features, step);
            } else if (op == 1) {
                findRightTriangleFeatures(next, current, 0, features, step);
            } else if (op == 2) {
                findRightTriangleFeaturesTally_UnOp(next, current, 0, features_tally, step);
            } else if (op == 3) {
                findRightTriangleFeaturesTally_Op2(next, current, 0, features_tally_list, step);
            } else if (op == 4) {
                findRightTriangleFeaturesTally(next, current, 0, features_tally, step);
            } 
            //findRightTriangleFeaturesTally(next, current, 0, features_tally, step);
            //findRightTriangleFeatures(next, current, 0, features, step);
            
        }
        if (rule == 90 || rule == 18 || rule == 22 || rule == 26 || rule == 82 || rule == 126) {
            //findEqualTriangleFeatures(next, current, 0, features, step);
        }
        //int rowAverageValue = rowAverage(next, current, 0, width);
        //average_tally[rowAverageValue]++;
        //totalCellAverage += rowAverageValue;
        current = next;
        //display recently computed cells. Save recently computed row so that all computed rows are saved and there are no 'ghost' rows.
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
    }
    //create dictionary to list the number of rows of a given percentage of state 0 cells, may be useful for classification.
    //parallelize this loop
    //average_tally[rowAverage(next, current, activeCells, 0, width)]++;
    //totalCellAverage /= steps;
    cout << "Average percentage of state 0 cells per row: " << totalCellAverage << "%\n";


    if (saveToFile) {
        img.close();
    }

    //printFeatures(features);
    printFeatureSizeCounts(features_tally, totalCellAverage, logFile);
    printFeatureSizeCounts(categorizeFeatureSizes(features), totalCellAverage, logFile);
    //printFeatureSizeCounts(categorizeFeatureSizes(features2));
    return features_tally;
}

vector<int> randomStart(int width, bool print = false) {
    //Random start using C++11 random library, may be more uniform than using rand() and allows for easier seeding for reproducibility.
    //Documentation for mersenne twister shows that engine is uniform.
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

//Add function that can create a random starting condition from a seed.
vector<int> seedStart(int width, int seed, bool print = false) {
    //Random start using C++11 random library, may be more uniform than using rand() and allows for easier seeding for reproducibility.
    std::mt19937 mersenne_engine(seed);  // Generates random integers
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
    //Change the way we generate random rows.
    //or investigate the way that the inital start is being generated. is it uniform?
    /*
    for(i = 0; i < width; i++) {
        if(rand() < 0.5) {
            vec[i] = 1;
        } else {
            vec[i] = 0;
        }
    }
    */
}



int main() {
    //consider stopping the simulation early if we hit the boundary of the width size
    const int width = 1000000; //increase to 7 figures.
    const int steps = 10000;

    //for(int rule = 1; rule <= 128; rule++) {

    const int rule = 110;  // Try 90, 110, 184, etc.
    vector<int> start(width, 0);
    //start[width / 2] = 1; //one in the middle
    //start[width-1] = 1;  //one on the edge
    //start = randomStart(width);
    int seed = rule;
    map<int, int> total_features;
    map<int, int> average_tally;
    for (int i = 0; i < 1; i++) {
        seed = i;
        start = seedStart(width, seed); 
        //might compare features and feature distribution for center in the middle v.s randomly generated starting conditions.
        //Need options to run the program without visuals.
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout<< "Simulating..." << std::endl;
        
        auto features_tally = imageSimulator(width, steps, rule, start, average_tally, 
                            "feature_log_" + to_string(rule) + "_" + to_string(seed) + ".txt", 
                            "feature_size_distribution.txt",
                            4);
        for(auto &p : features_tally) {
            total_features[p.first] += p.second;
        }
        //total_features.merge(features_tally);
        std::cout << "Simulated." << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

        std::ofstream img;
        img.open("feature_log_" + to_string(rule) + "_" + to_string(seed) + ".txt", std::ios_base::app);
        img << "Seed: " << seed << "\n";
        img << "Total execution time: " << elapsed.count() << " seconds\n";
        img << "Width: " << width << "\n";
        img << "Steps: " << steps << "\n";
        img << "==============================\n";
        img.close();
    }
    //investigate moving the code responsible for compiling the data across multiple runs into a separate function
    //create csv file from total_features dictionary, with two columns, feature size and count.
    std::ofstream csv;
    csv.open("total_feature_size_distribution " + to_string(rule) + ".csv");
    csv << "Feature Size,Count\n";
    for (const auto& entry : total_features) {
        csv << entry.first << "," << entry.second << "\n";
    }
    csv.close();

    //save average tally to csv file, with two columns, average percentage of state 0 cells and count.
    saveAverageTally(average_tally, "average_tally_" + to_string(rule) + ".csv");
    
    //}
    return 0;

}

//Profile the code with Visual Studio Performance Profiler.
//Run a bunch of simulations with different seeds. Average feature size distribution.
//Save the feature size distributions to a csv file.
//Why does it take so long to run with a profiler? 8x slower.

//make sure that debug flags are used when compile so that the profiler can point me to actual function names and lines of code, rather than just assembly instructions. This will allow me to identify which parts of the code are taking the most time and optimize them.

//Test various versions of the code and compare the performance of the variations. Think of this as an experimental and form a hypothesis and 
//collect data to support or refute the hypothesis. For example, I could hypothesize that using a concurrent hash map for the features tally will reduce contention and improve performance compared to using a critical section. I could implement both versions of the code, run them with the profiler, and compare the results to see if the hypothesis is supported.
//Find some means of making a performance measurement.
//If we make change x the profile results show y etc.
//When we check the version that uses the previous data structure make sure that we get memory results in the profiler.
//check total memory allocated. Check cache miss rate or DRAM bandwith consumed/ data draw.

//Replace the map with an array.
