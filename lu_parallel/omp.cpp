#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <windows.h>
#include <string>
#include <time.h>
#include <immintrin.h>

using namespace std;
const int maxN = 1024;
const int THREAD_NUM = 4;
string filename = "result_schedule2.txt";
bool ifshow = 0;

long long head, tail, freq;
long long thread_head[THREAD_NUM], thread_tail[THREAD_NUM];

float A[maxN][maxN];
float B[maxN][maxN];
float C[maxN][maxN];
float D[maxN][maxN];
float E[maxN][maxN];
float F[maxN][maxN];
float G[maxN][maxN];


void init_matrix(int n, float a[][maxN]) {
	srand((int)time(0));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			//a[i][j] = B[i][j] = C[i][j] = (float)rand()+1.0;
			a[i][j] = (float)rand() + 1.0;
		}
	}
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

//A
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
//B:bad performance:frequently create threads
void omp_simple1(int n, float a[][maxN]) {
	int i, j, k;
	for (k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
#pragma omp parallel for num_threads(THREAD_NUM)\
default(none) private(i,j) shared(a,n,k)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
//C
void omp_simple2(int n, float a[][maxN]) {
	int i, j, k;
	for (k = 0; k < n; k++) {
#pragma omp parallel for num_threads(THREAD_NUM)\
default(none) private(j) shared(a,n,k)
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
#pragma omp parallel for num_threads(THREAD_NUM)\
default(none) private(i,j) shared(a,n,k)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
//D:optimized from simple1
void omp_simple3(int n, float a[][maxN]) {
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM)\
default(none) shared(a,n) private(i,j,k)
	for (k = 0; k < n; k++) {
#pragma omp for
		for (j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
#pragma omp for
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
//E schedule 
void omp_guide(int n, float a[][maxN]) {
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM)\
default(none) private(i,j,k) shared(a,n) 
	for (k = 0; k < n; k++) {
#pragma omp for
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
//#pragma omp barrier
#pragma omp for schedule(guided)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
void omp_dynamic(int n, float a[][maxN]) {
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM)\
default(none) private(i,j,k) shared(a,n) 
	for (k = 0; k < n; k++) {
#pragma omp for
		for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		//#pragma omp barrier
#pragma omp for schedule(dynamic,50)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
//F
void omp_sse(int n, float a[][maxN]) {
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM)\
default(none) shared(a,n) private(i,j,k)
	{
		__m128 t1, t2, t3, t4;
		for (int k = 0; k < n; k++) {
			float pack[4] = { a[k][k],a[k][k], a[k][k], a[k][k] };
			t2 = _mm_loadu_ps(pack);
			int j = k + 1;
			for (; j < n - 3; j += 4) {
				t1 = _mm_loadu_ps(a[k] + j);
				t1 = _mm_div_ps(t1, t2);
				_mm_storeu_ps(a[k] + j, t1);
			}
//#pragma omp for
			for (; j < n; j++) {
				a[k][j] = a[k][j] / a[k][k];
			}
			a[k][k] = 1.0; 
#pragma omp for nowait
			for (int i = k + 1; i < n; i++) {
				float pack[4] = { a[i][k],a[i][k], a[i][k], a[i][k] };
				t3 = _mm_loadu_ps(pack);
				j = k + 1;
				for (; j + 3 < n; j += 4) {
					t1 = _mm_loadu_ps(a[i] + j);
					t4 = _mm_loadu_ps(a[k] + j);
					t4 = _mm_mul_ps(t3, t4);
					t1 = _mm_sub_ps(t1, t4);
					_mm_storeu_ps(a[i] + j, t1);
				}
				for (; j < n; j++) {
					a[i][j] = a[i][j] - a[i][k] * a[k][j];
				}
				a[i][k] = 0;
			}

		}
	}
}
//G
void omp_simple4(int n, float a[][maxN]) {
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM)\
default(none) shared(a,n)
	for (int k = 0; k < n; k++) {
#pragma omp master
		{for (int j = k + 1; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0; }
#pragma omp barrier
#pragma omp for nowait
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
int main() {
	init_matrix(maxN, A);
	memcpy(&B, &A, sizeof(A));
	memcpy(&C, &A, sizeof(A));
	memcpy(&D, &A, sizeof(A));
	memcpy(&E, &A, sizeof(A));
	memcpy(&F, &A, sizeof(A));
	memcpy(&G, &A, sizeof(A));

	if (ifshow)
		showResult(A);

	ofstream writefile(filename, ios::app);
	float time;

	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu(maxN, A);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "lu  " << time << "ms" << endl;
	writefile << time << '\t';

	////scheme A
	//QueryPerformanceCounter((LARGE_INTEGER *)&head);
	//omp_simple1(maxN, B);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	//time = (tail - head) * 1000.0 / freq;
	//cout << "simple1  " << time << "ms" << endl;
	//writefile << time << '\t';

	////scheme AC
	//QueryPerformanceCounter((LARGE_INTEGER *)&head);
	//omp_simple4(maxN, G);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	//time = (tail - head) * 1000.0 / freq;
	//cout << "simple4  " << time << "ms" << endl;
	//writefile << time << '\t';

	////scheme B
	//QueryPerformanceCounter((LARGE_INTEGER *)&head);
	//omp_simple2(maxN, C);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	//time = (tail - head) * 1000.0 / freq;
	//cout << "simple2  " << time << "ms" << endl;
	//writefile << time << '\t';
	//
	////scheme BC
	//QueryPerformanceCounter((LARGE_INTEGER *)&head);
	//omp_simple3(maxN, D);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	//time = (tail - head) * 1000.0 / freq;
	//cout << "simple3  " << time << "ms" << endl;
	//writefile << time << '\t';

	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	omp_simple3(maxN, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "static  " << time << "ms" << endl;
	writefile << time << '\t';

	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	omp_guide(maxN, C);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "guide  " << time << "ms" << endl;
	writefile << time << '\t';

	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	omp_dynamic(maxN, E);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "dynamic  " << time << "ms" << endl;
	writefile << time << '\t';

	/*QueryPerformanceCounter((LARGE_INTEGER *)&head);
	omp_sse(maxN, F);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	time = (tail - head) * 1000.0 / freq;
	cout << "sse  " << time << "ms" << endl;
	writefile << time << '\t';*/

	writefile << '\n';
	writefile.close();
	if (ifshow) {
		showResult(A);
		showResult(C);
	}

	//if (checkCorrect(A, B)) {
	//	cout << "correct B" << endl;
	//}
	//if (checkCorrect(A, C)) {
	//	cout << "correct C" << endl;;
	//}
	//if (checkCorrect(A, D)) {
	//	cout << "correct D" << endl;;
	//}
	//if (checkCorrect(A, E)) {
	//	cout << "correct E" << endl;;
	//}
	//if (checkCorrect(A, F)) {
	//	cout << "correct F" << endl;;
	//}
	//if (checkCorrect(A, G)) {
	//	cout << "correct G" << endl;;
	//}
}