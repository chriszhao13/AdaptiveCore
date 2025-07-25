
#include "../include/graph.h"

bool degComp(const pair<unsigned int, unsigned int> &lhs, const pair<unsigned int, unsigned int> &rhs)
{
    return lhs.second > rhs.second;
}

bool Graph::readSerialized(string input_file)
{
    

    ifstream file;

    file.open(input_file);

    if (file)
    {

        // cout << "Reading serialized file... " << endl;

        file >> V;
        file >> E;
        degrees = new unsigned int[V];
        neighbors_offset = new unsigned int[V + 1];
        neighbors = new unsigned int[E];
        for (int i = 0; i < V; i++)
            file >> degrees[i];

        for (int i = 0; i < V + 1; i++)
            file >> neighbors_offset[i];

        for (unsigned int i = 0; i < E; i++)
            file >> neighbors[i];

        file.close();
        return true;
    }
    else
    {
        cout << input_file << " : File couldn't open" << endl;
    }

    return false;
}

void Graph::writeSerialized(string input_file)
{
    input_file = input_file.substr(input_file.find_last_of("/") + 1);

    ofstream file;
    file.open(string(OUTPUT_LOC) + string("serialized-") + input_file);
    if (file)
    {
        file << V << endl;
        file << E << endl;
        for (int i = 0; i < V; i++)
            file << degrees[i] << endl;
        for (int i = 0; i < V + 1; i++)
            file << neighbors_offset[i] << ' ';
        for (int i = 0; i < E; i++)
            file << neighbors[i] << ' ';
        file.close();
    }
    else
    {
        cout << string("serialized-") + input_file << " : File couldn't open" << endl;
    }
}
Graph::Graph()
{
    // default constructor
}
void Graph::readFile(string input_file)
{

    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(DS_LOC + input_file);
    if (!infile)
    {
        cout << "load graph file failed " << endl;
        exit(-1);
    }

    unsigned int s, t, value;

    string line;
    vector<pair<unsigned int, unsigned int>> lines;

    V = 0;
    int s_ = 0;
    unsigned int num_nodes, num_edges;

    std::getline(infile, line);
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> num_nodes >> num_nodes >> num_edges;

    while (std::getline(infile, line))
    {

        std::istringstream iss(line);
        iss >> s >> t >> value;
        s = s - 1;
        t = t - 1;
        if (s_ != s)
        {
            // cout<<"Geting "<<s<<"\r";
            s_ = s;
        }
        if (s == t)
            continue;
        V = max(s, V);
        V = max(t, V);
        lines.push_back({s, t});
    }
    infile.close();
    cout << endl;
    V++; // vertices index starts from 0, so add 1 to number of vertices.
    vector<set<unsigned int>> ns(V);

    for (auto &p : lines)
    {
        ns[p.first].insert(p.second);
        ns[p.second].insert(p.first);
    }

    lines.clear();

    cout << "CSR..." << endl;

    degrees = new unsigned int[V];
    for (int i = 0; i < V; i++)
    {
        degrees[i] = ns[i].size();
    }

    neighbors_offset = new unsigned int[V + 1];
    neighbors_offset[0] = 0;
    partial_sum(degrees, degrees + V, neighbors_offset + 1);

    E = neighbors_offset[V];
    neighbors = new unsigned int[E];

#pragma omp parallel for
    for (int i = 0; i < V; i++)
    {
        auto it = ns[i].begin();
        for (int j = neighbors_offset[i]; j < neighbors_offset[i + 1]; j++, it++)
            neighbors[j] = *it;
    }
    writeSerialized(input_file);
}

void Graph::writeKCoreToDisk(std::string file)
{
    // writing kcore in json dictionary format
    std::ofstream out(OUTPUT_LOC + string("pkc-kcore-") + file);

    out << "{ ";

    for (unsigned long long int i = 0; i < V; ++i)

        if (degrees[i] != 0)
            out << '"' << i << '"' << ": " << degrees[i] << ", " << endl;
    out.seekp(-3, ios_base::end);
    out << " }";
    out.close();
}

Graph::Graph(std::string input_file)
{
    auto start = chrono::steady_clock::now();
    if (readSerialized(input_file))
    {
        cout << "Read Yes! ";
        return;
    }
    auto end = chrono::steady_clock::now();
    cout << "File Loaded in: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << endl;
}

Graph::~Graph()
{

    // cout << "Deallocated... " << endl;

    delete[] neighbors;
    delete[] neighbors_offset;
    delete[] degrees;
}