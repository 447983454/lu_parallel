#pragma comment(lib,"pthreadVC2.lib")

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <vector>
#include <time.h>
#include <immintrin.h>
#include <windows.h>
#include <pthread.h>
#include <semaphore.h>
#include <string>

using namespace std;

const int maxN = 1024;
const int THREAD_NUM = 4;
string filename = "t4_1024data.txt";

long long head, tail, freq;
long long thread_head[THREAD_NUM], thread_tail[THREAD_NUM];

float A[maxN][maxN];
float B[maxN][maxN];
float C[maxN][maxN];
float D[maxN][maxN];
float E[maxN][maxN];
float F[maxN][maxN];
float G[maxN][maxN];

pthread_barrier_t barrier_start;
pthread_barrier_t barrier_stop;

sem_t sem_k;

typedef struct param_struct_A{
	int threadId;
	int row_num;
	float(* matrix)[maxN];
} threadParm_t;


void init_matrix(int n, float a[][maxN]) {
	srand((int)time(0));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			//a[i][j] = B[i][j] = C[i][j] = (float)rand()+1.0;
			a[i][j] = (float)rand() + 1.0;
		}
	}
}
void lu(int n, float a[][maxN]) {
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

// X
void* pthread_simple_fun(void *parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int id = p->threadId;
	int k = p->row_num;
	float(* a)[maxN] = p->matrix;
	int i = k + 1;
	for (; i < maxN; i++) {
		if (i%THREAD_NUM == id)
			break;
	}
	for (; i < maxN; i += THREAD_NUM) {
		for (int j = k + 1; j < maxN; j++) {
			a[i][j] = a[i][j] - a[i][k] * a[k][j];
		}
		a[i][k] = 0;
	}
	return NULL;
}
void lu_pthread_simple(int n, float a[][maxN]) {
//void lu_pthread(int n, float** a) {
	pthread_t thread[THREAD_NUM];
	threadParm_t threadParm[THREAD_NUM];
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = 0; i < THREAD_NUM; i++) {
			threadParm[i].row_num = k;
			threadParm[i].threadId = i;
			threadParm[i].matrix = a;
			pthread_create(&thread[i], NULL, pthread_simple_fun, &threadParm[i]);
		}
		for (int i = 0; i < THREAD_NUM; i++) {
			pthread_join(thread[i], NULL);
		}
	}
}

void* pthread_barrier_fun(void *parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int id = p->threadId;
	float(*a)[maxN] = p->matrix;
	//QueryPerformanceCounter((LARGE_INTEGER *)&thread_head[id]);

	for (int k = 0; k < maxN; k++) {
		pthread_barrier_wait(&barrier_start);
		int i = k + 1;
		for (; i < maxN; i++) {
			if (i%THREAD_NUM == id)
				break;
		}
		for (; i < maxN; i+=THREAD_NUM) {
			for (int j = k + 1; j < maxN; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
		pthread_barrier_wait(&barrier_stop);
	}
	//QueryPerformanceCounter((LARGE_INTEGER *)&thread_tail[id]);
	//cout << "lu_pthread_barrier_"<< id << ":" << (tail - head) * 1000.0 / freq << "ms" << endl;
	return NULL;
}
void lu_pthread_barrier(int n, float a[][maxN]) {
	pthread_t thread[THREAD_NUM];
	threadParm_t threadParm[THREAD_NUM];
	pthread_barrier_init(&barrier_start, NULL, THREAD_NUM+1);
	pthread_barrier_init(&barrier_stop, NULL, THREAD_NUM+1);

	for (int i = 0; i < THREAD_NUM; i++) {
		threadParm[i].row_num = NULL;
		threadParm[i].threadId = i;
		threadParm[i].matrix = a;
		pthread_create(&thread[i], NULL, pthread_barrier_fun, &threadParm[i]);
	}
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		pthread_barrier_wait(&barrier_start);
		pthread_barrier_wait(&barrier_stop);
	}
	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], NULL);
	}

	pthread_barrier_destroy(&barrier_start);
	pthread_barrier_destroy(&barrier_stop);

}

void* pthread_barrier2_fun(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int id = p->threadId;
	float(*a)[maxN] = p->matrix;
	//QueryPerformanceCounter((LARGE_INTEGER *)&thread_head[id]);

	for (int k = 0; k < maxN; k++) {
		pthread_barrier_wait(&barrier_start);
		int i = k + 1;
		for (; i < maxN; i++) {
			if (i%THREAD_NUM == id)
				break;
		}
		for (; i < maxN; i += THREAD_NUM) {
			for (int j = k + 1; j < maxN; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
			if (i == k + 1) {
				sem_post(&sem_k);
			}
		}
	}
	//QueryPerformanceCounter((LARGE_INTEGER *)&thread_tail[id]);
	//cout << "lu_pthread_barrier_" << id << ":" << (tail - head) * 1000.0 / freq << "ms" << endl;
	return NULL;
}
void lu_pthread_barrier2(int n, float a[][maxN]) {
	pthread_t thread[THREAD_NUM];
	threadParm_t threadParm[THREAD_NUM];
	pthread_barrier_init(&barrier_start, NULL, THREAD_NUM + 1);

	for (int i = 0; i < THREAD_NUM; i++) {
		threadParm[i].row_num = NULL;
		threadParm[i].threadId = i;
		threadParm[i].matrix = a;
		pthread_create(&thread[i], NULL, pthread_barrier2_fun, &threadParm[i]);
	}
	sem_init(&sem_k, 0, 1);
	for (int k = 0; k < n; k++) {
		sem_wait(&sem_k);
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		pthread_barrier_wait(&barrier_start);
	}
	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], NULL);
	}

	pthread_barrier_destroy(&barrier_start);

}

void* pthread_col_fun(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int id = p->threadId;
	float(*a)[maxN] = p->matrix;

	for (int k = 0; k < maxN; k++) {
		int j = k + 1;
		for (; j < maxN; j++) {
			if (j%THREAD_NUM == id)
				break;
		}
		for (; j < maxN; j+=THREAD_NUM) {
			//a[k][j] = a[k][j] / a[k][k];
			a[j][k] = a[j][k] / a[k][k];
			for (int i = k + 1; i < maxN; i++) {
				//a[i][j] = a[i][j] - a[i][k] * a[k][j];
				a[j][i] = a[j][i] - a[k][i] * a[j][k];
			}
		}
		pthread_barrier_wait(&barrier_start);
		if (id == 0) {
			a[k][k] = 1.0;
			for (int i = k + 1; i < maxN; i++) {
				a[k][i] = 0;
			}
		}
	}
	return NULL;
}
void* pthread_col_block_fun(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int id = p->threadId;
	float(*a)[maxN] = p->matrix;

	for (int k = 0; k < maxN; k++) {
		int j = k + 1;

		int start, next;
		int bound = ceil((maxN - j + 1) / float(THREAD_NUM));
		start = j + (bound)*id;
		next = min((j + (bound)*(id + 1)),maxN);
		for (j=start; j < next; j ++) {
			a[j][k] = a[j][k] / a[k][k];
			for (int i = k + 1; i < maxN; i++) {
				a[j][i] = a[j][i] - a[k][i] * a[j][k];
			}
		}
		pthread_barrier_wait(&barrier_start);
		if (id == 0) {
			a[k][k] = 1.0;
			for (int i = k + 1; i < maxN; i++) {
				a[k][i] = 0;
			}
		}
	}
	return NULL;
}
void* pthread_col_block_sse_fun(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int id = p->threadId;
	float(*a)[maxN] = p->matrix;
	__m128 t1, t2, t3, t4;

	for (int k = 0; k < maxN; k++) {
		int j = k + 1;

		int start, next;
		int bound = ceil((maxN - j + 1) / float(THREAD_NUM));
		start = j + (bound)*id;
		next = min((j + (bound)*(id + 1)), maxN);

		for (j = start; j < next; j++) {
			a[j][k] = a[j][k] / a[k][k];

			float pack[4] = { a[j][k],a[j][k], a[j][k], a[j][k]};
			t3 = _mm_loadu_ps(pack);
			int i = k + 1;
			for (; i + 3 < maxN; i += 4) {
				t1 = _mm_loadu_ps(a[j] + i);
				t4 = _mm_loadu_ps(a[k] + i);
				t4 = _mm_mul_ps(t3, t4);
				t1 = _mm_sub_ps(t1, t4);
				_mm_storeu_ps(a[j] + i, t1);
			}
			for (; i < maxN; i++) {
				a[j][i] = a[j][i] - a[k][i] * a[j][k];
			}
		}

		pthread_barrier_wait(&barrier_start);
		if (id == 0) {
			a[k][k] = 1.0;
			for (int i = k + 1; i < maxN; i++) {
				a[k][i] = 0;
			}
		}
	}
	return NULL;
}
void lu_pthread_col(int n, float a[][maxN],void*(*fun)(void*)) {
	pthread_t thread[THREAD_NUM];
	threadParm_t threadParm[THREAD_NUM];
	pthread_barrier_init(&barrier_start, NULL, THREAD_NUM);
	for (int i = 0; i < THREAD_NUM; i++) {
		threadParm[i].row_num = NULL;
		threadParm[i].threadId = i;
		threadParm[i].matrix = a;
		pthread_create(&thread[i], NULL,fun, &threadParm[i]);
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], NULL);
	}
	pthread_barrier_destroy(&barrier_start);
}

bool checkCorrect(float a[][maxN], float b[][maxN]) {
	for (int i = 0; i < maxN; i++) {
		for (int j = 0; j < maxN; j++) {
			if (a[i][j] != b[i][j]) {
				return false;
			}
		}
	}
	return true;
}
void showResult(float a[][maxN]) {
	for (int i = 0; i < maxN; i++) {
		for (int j = 0; j < maxN; j++) {
			cout << setw(10) << a[i][j];
			cout << " ";
		}
		cout << endl;
	}
	cout << endl;
}
void copy_trans(float src[][maxN], float dst[][maxN]) {
	for (int i = 0; i < maxN; i++) {
		for (int j = 0; j < maxN; j++) {
			dst[j][i] = src[i][j];
		}
	}
}
int main(int argc, char *argv[])
{
	init_matrix(maxN, A);
	memcpy(&B, &A, sizeof(A));
	memcpy(&C, &A, sizeof(A));
	copy_trans(A, D);
	memcpy(&E, &D, sizeof(D));
	memcpy(&F, &D, sizeof(D));

	//showResult(A);
	//showResult(F);
	ofstream writefile(filename, ios::app);
	float time;

	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu(maxN, A);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "lu  " << time << "ms" << endl;
	writefile << time << '\t';

	////unfavorable
	//QueryPerformanceCounter((LARGE_INTEGER *)&head);
	//lu_pthread_simple(maxN, B);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	//cout << "lu_pthread_simple  " << (tail - head) * 1000.0 / freq << "ms" << endl;

	//Scheme A
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu_pthread_barrier(maxN, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "lu_pthread_barrier  " << time << "ms" << endl;
	writefile << time << '\t';

	//Scheme B
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu_pthread_barrier2(maxN, C);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "lu_pthread_barrier2  " << time << "ms" << endl;
	writefile << time << '\t';

	//Scheme C
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu_pthread_col(maxN, D, pthread_col_fun);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "lu_pthread_col  " << time << "ms" << endl;
	copy_trans(D, B);
	memcpy(&D, &B, sizeof(D));
	writefile << time << '\t';

	//Scheme D
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu_pthread_col(maxN, E, pthread_col_block_fun);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "lu_pthread_col_block  " << time << "ms" << endl;
	copy_trans(E, B);
	memcpy(&E, &B, sizeof(E));
	writefile << time << '\t';

	//Scheme E
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu_pthread_col(maxN, F, pthread_col_block_sse_fun);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "lu_pthread_col_block_sse  " << time << "ms" << endl;
	copy_trans(F, B);
	memcpy(&F, &B, sizeof(F));
	writefile << time << '\t';

	writefile << '\n';
	//showResult(A);
	//showResult(F);

	if (checkCorrect(A, F)) {
		cout << "correct answer";
	}
	else
		cout << "wrong answer";

}
