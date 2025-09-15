__global__ void addVec(float* a, float* b, float* c, int N);

void addVec_cpu(float* a, float* b, float* c, int N);

void init_vectors(float** a, float** b, float** c, int N);