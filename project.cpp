/*  
*   CITS3402: High Performance Computing, Project 1
*   
*   Project by: Jeremiah Pinto (21545883), Jainish Pithadiya (21962504)
*/

// Header Files
#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int TIME_STEPS = 100;
int N_DIMENSIONS;
int MAX_THREADS = omp_get_max_threads();

// Function that outputs the given grid to a file with no extension.
// The file name is based on the grid's relative time step
void GridToFile(bool *GOL[], bool isFinalGrid = false, bool isIntermediate = false)
{
    string file_name = to_string(N_DIMENSIONS) + "x" + to_string(N_DIMENSIONS) + "_" + (isFinalGrid ? "final" : (isIntermediate ? "intermediate" : "initial"));
    ofstream file;
    file.open (file_name);
    stringstream buf;

    for(int i = 0; i < N_DIMENSIONS; i++)
    {
        for(int j = 0; j < N_DIMENSIONS; j++)
        {    
            buf << GOL[i][j] << " ";
        }
        buf << "\n";
    }
    file << buf.rdbuf();;
    file.close();
}

// The logic based on Conway's Game of Life.
// We are only checking immediate neighbours (von Neumann neighbourhood)
bool SetCellStatus(bool** GOL, int i, int j)
{
	int neighbours = 0;
	if(GOL[((i+1) % N_DIMENSIONS + N_DIMENSIONS) % N_DIMENSIONS][j]) neighbours++;
	if(GOL[((i-1) % N_DIMENSIONS + N_DIMENSIONS) % N_DIMENSIONS][j]) neighbours++;
	if(GOL[i][((j-1) % N_DIMENSIONS + N_DIMENSIONS) % N_DIMENSIONS]) neighbours++;
	if(GOL[i][((j+1) % N_DIMENSIONS + N_DIMENSIONS) % N_DIMENSIONS]) neighbours++;
	return (neighbours == 3 || (GOL[i][j] ? (neighbours > 1 && neighbours != 4) : false));
}

// Sequential run through of the Game of Life.
// Returns time taken on completion
auto SequentialRunthrough(bool *GOL[], bool** nextState)
{
    auto start = chrono::steady_clock::now();

	for(int time_step_number = 0; time_step_number < TIME_STEPS; time_step_number++)
	{
		for(int i = 0; i < N_DIMENSIONS; i++)
			for(int j = 0; j < N_DIMENSIONS; j++)
				nextState[i][j] = SetCellStatus(GOL,i,j);

        for(int i = 0; i < N_DIMENSIONS; i++)
            for(int j = 0; j < N_DIMENSIONS; j++)
                GOL[i][j] = nextState[i][j];
	}

    auto end = chrono::steady_clock::now();

    return end - start;
}

// Parallel run through of the Game of Life.
// Returns time taken on completion
// EXTRA FEATURE: Restricts number of threads used based on user input.
// EXTRA OPTIMIZATIONS : Scheduling implemented
// If number of threads not defined by user, uses all threads available
auto ParallelRunthrough(bool *GOL[], bool** nextState)
{
    omp_set_num_threads(MAX_THREADS);

    auto start = chrono::steady_clock::now();

    #pragma omp parallel
    {
        for(int time_step_number = 0; time_step_number < TIME_STEPS; time_step_number++)
        {
            #pragma omp for schedule(dynamic)
            for(int i = 0; i < N_DIMENSIONS; i++)
            {
                for(int j = 0; j < N_DIMENSIONS; j++)
                {
                    nextState[i][j] = SetCellStatus(GOL,i,j);
                }
            }

            #pragma omp for schedule(static)
            for(int i = 0; i < N_DIMENSIONS; i++)
            {
                for(int j = 0; j < N_DIMENSIONS; j++)
                {
                    GOL[i][j] = nextState[i][j];
                }
            }
        }
    }

    auto end = chrono::steady_clock::now();

    return end - start;
}

// Function to clear all dynamic memory used in the program
void ClearMemory(bool **GOL, bool **GOL2, bool **nextState) 
{
    for(int i = 0; i < N_DIMENSIONS; i++)
    {
        free(GOL[i]);
        free(GOL2[i]);
        free(nextState[i]);
    }

    free(GOL);
    free(GOL2);
    free(nextState);
}

int main (int argc, char *argv[]) 
{
    // Set initial grid size, with default to 128x128
    if (argc < 2)
    {
        N_DIMENSIONS = 128;
    }
    else
    {
        int temp = stoi(argv[1]);
        
        // Sets grid size based on user input
        // Terminates program if user input size is not a valid one
        if (temp == 128 || temp == 256 || temp == 512 || temp == 1024 || temp == 2048)
        {
            N_DIMENSIONS = temp;
        }
        else
        {
            cerr << "ERROR: Invalid Grid Size\n" << "Sizes Allowed: 128, 256, 512, 1024, 2048. "<< temp <<" Provided"<< "\n";
            cerr << "If on Windows, ensure format is: project <grid-size> <number-of-threads> -<time-steps> g\n";
            cerr << "If on macOS, ensure format is: ./project <grid-size> <number-of-threads> -<time-steps> g\n";
            return EXIT_FAILURE;
        }

        // Checks if user has a restriction on number of threads to be used
        // Terminates program if user has enter an invalid thread count
        // compared to current machine running the program
        if (argc > 2 && argv[2][0] != 'g' && argv[2][0] != '-')
        {
            temp = stoi(argv[2]);

            if (temp > MAX_THREADS || temp == 0) {
                cerr << "Number of threads used are invalid. \nMaximum number of threads available are " << MAX_THREADS << "\n" << temp <<" Provided\n";
                cerr << "If on Windows, ensure format is: project <grid-size> <number-of-threads> -<time-steps> g\n";
                cerr << "If on macOS, ensure format is: ./project <grid-size> <number-of-threads> -<time-steps> g\n";
                return EXIT_FAILURE;
            }
            else
            {
                MAX_THREADS = temp;
            }
        }
    }
    // add -TIME_STEP tag to set timestep value
    for (char** args = argv; *args; ++args)
    {
        char* arg = *args;
        if (arg[0] == '-')
        {
            arg++;
            TIME_STEPS = stoi(arg);

            if (TIME_STEPS < 100)
            {
                cerr << "Invalid number of time steps. Must be more that 100\n";
                return EXIT_FAILURE;
            }
        }
    }

    // Generate initial grid with random numbers
    srand(time(0));
    
    bool **GOL; 
    bool **GOL2;
    bool **nextState; 

    GOL = (bool **) malloc(sizeof(bool *)* N_DIMENSIONS);
    GOL2 = (bool **) malloc(sizeof(bool *)* N_DIMENSIONS);
    nextState = (bool **) malloc(sizeof(bool *)* N_DIMENSIONS);

    if(GOL == NULL || nextState == NULL || GOL2 == NULL)
    {
        cerr << "Memory allocation failed\n";
        return EXIT_FAILURE;
    }

    for(int i = 0; i < N_DIMENSIONS; i++)
    {
        GOL[i] = (bool *) malloc(sizeof(bool)* N_DIMENSIONS);
        nextState[i] = (bool *) malloc(sizeof(bool)* N_DIMENSIONS);
        GOL2[i] = (bool *) malloc(sizeof(bool)* N_DIMENSIONS);

        if(GOL[i] == NULL || nextState[i] == NULL || GOL2[i] == NULL)
        {
            cerr << "Memory allocation failed\n";
            return 0;
        }

        for(int j = 0; j < N_DIMENSIONS; j++)
        {   
            bool randbool = rand() & 1;
            GOL[i][j] = randbool;
            GOL2[i][j] = randbool;
            nextState[i][j] = randbool;
        }
    }

    // Outputs initial grid to file if 'g' is in commandline arguements
    if(argc > 1 && argv[argc - 1][0] == 'g')
        GridToFile(GOL);

    // Get time of Sequential runthrough
    auto sequentialTime = SequentialRunthrough(GOL, nextState);

    // Get time of Parallel runthrough
    auto parallelTime   = ParallelRunthrough(GOL2, nextState);

    // Outputs grid after 100 time steps to file if 'g' is in commandline arguements
    if(argc > 1 && argv[argc - 1][0] == 'g')
        GridToFile(GOL,true);

    // Displays the time for Sequential and Parallel run throughs and the threads used
    cout << "Sequential run time for Game of Life of grid size " << N_DIMENSIONS << "x" << N_DIMENSIONS << " is: " << chrono::duration <double, milli> (sequentialTime).count() << "ms\n";
    cout << "Parallel run time for Game of Life of grid size "   << N_DIMENSIONS << "x" << N_DIMENSIONS << " is: " << chrono::duration <double, milli> (parallelTime).count()   << "ms\n";
    
    cout << "\nTime steps used: " << TIME_STEPS << '\n';
    cout << "Number of threads used for parallel runthrough: " << MAX_THREADS << '\n';
    
    // Clear all dynamic memory before program terminates to prevent memory leaks
    ClearMemory(GOL, GOL2, nextState);

    return EXIT_SUCCESS;
}

