__global__ void addVec(float* a, float* b, float* c, int N);

void addVec_cpu(float* a, float* b, float* c, int N);

void addVec_gpu(float* a, float* b, float* c, int N);

enum class VecType {
    Host,
    Device
};

template<typename T>
class Vectors {
private:
    struct Vec {
        T* ptr;
        VecType type;
    };
    std::vector<Vec> vectors;
    size_t N;
    size_t size;

public:
    Vectors(int N, int countVecs);
    Vectors(int N, int countVecs, bool cudaDevice);
    
    ~Vectors();
    
    void init_vectors(std::initializer_list<T*> pointers);
    void copy_device_vectors(std::initializer_list<T*> host_pointers,
                std::initializer_list<T*> device_pointers, 
                 bool fromDevice);
    std::vector<T*> getVectors(int inds);
};
