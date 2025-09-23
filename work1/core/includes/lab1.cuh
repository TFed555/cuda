__global__ void addVec(float* a, float* b, float* c, int N);

void addVec_cpu(float* a, float* b, float* c, int N);

void addVec_gpu(float* a, float* b, float* c, int N);

void init_vectors(float** a, float** b, float** c, int N);

void copy_vectors(float* host_a, float* host_b, float* host_c, float** device_a, float** device_b, float** device_c, int N);

void free_vectors(float* a, float* b, float* c);

void cudafree_vectors(float* device_a, float* device_b, float* device_c);