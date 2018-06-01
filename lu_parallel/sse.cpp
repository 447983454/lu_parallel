#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include <fstream> 
using namespace std;

const int maxN = 1024;

long long head, tail, freq;        //timers
double result1, result2;

// float C[maxN][maxN];
float A[maxN][maxN];
_MM_ALIGN16 float B[maxN][maxN];


//
//int powof2[32] =
//{
//	1,           2,           4,           8,         16,          32,
//	64,         128,         256,         512,       1024,        2048,
//	4096,        8192,       16384,       32768 
//};


void init(int n, float a[][maxN]) {
	srand((int)time(0));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			//a[i][j] = B[i][j] = C[i][j] = (float)rand()+1.0;
			a[i][j] = (float)rand()+1.0;
			B[i][j] = a[i][j];
		}
	}
}
//void optimized_init(int *n, float a[][maxN]) {
//	int num = *n;
//	if (num&(num - 1)) {
//		//not 2^n
//		int count = 0;
//		while (num) {
//			count++;
//			num >>= 1;
//		}
//		num = (*n - powof2[count - 1]) < (powof2[count] - *n) ? powof2[count - 1] : powof2[count];
//	}
//}

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
void lu_sse(int n, float a[][maxN]) {
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < n; k++) {
		float pack[4] = { a[k][k],a[k][k], a[k][k], a[k][k] };
		t2 = _mm_loadu_ps(pack);
		//t2 = _mm_load1_ps(a[k] + k);
		int j = k + 1;
		for (; j + 3 < n; j += 4) {
			t1 = _mm_loadu_ps(a[k] + j);
			t1 = _mm_div_ps(t1, t2);
			_mm_storeu_ps(a[k]+j, t1);
		}
		for (; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			float pack[4] = { a[i][k],a[i][k], a[i][k], a[i][k] };
			t3 = _mm_loadu_ps(pack);
			//t3 = _mm_load1_ps(a[i] + k);
			j = k + 1;
			for (; j + 3 < n; j += 4) {
				t1 = _mm_loadu_ps(a[i] + j);
				t4 = _mm_loadu_ps(a[k] + j);
				t4 = _mm_mul_ps(t3, t4);
				t1 = _mm_sub_ps(t1, t4);
				_mm_storeu_ps(a[i]+j, t1);
			}
			for (; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
		
	}
}
void lu_sse_align(int n, float a[][maxN]) {
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < n; k++) {
		_MM_ALIGN16 float pack[4] = { a[k][k],a[k][k], a[k][k], a[k][k] };
		t2 = _mm_load_ps(pack);
		//t2 = _mm_load1_ps(a[k] + k);
		int j = k + 1;
		if(j % 4!=0){
			int count = (j / 4 + 1) * 4 < n ? (j / 4 + 1) * 4 : n;
			for (; j < count; j++) {
				a[k][j] = a[k][j] / a[k][k];
			}
	/*		for (int count = 4 - j % 4; count > 0 && j < n; count--, j++) 
				a[k][j] = a[k][j] / a[k][k];*/
		}
		for (; j + 3 < n; j += 4) {
			t1 = _mm_load_ps(a[k] + j);
			t1 = _mm_div_ps(t1, t2);
			_mm_store_ps(a[k] + j, t1);
		}
		for (; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;

		for (int i = k + 1; i < n; i++) {
			_MM_ALIGN16 float pack[4] = { a[i][k],a[i][k], a[i][k], a[i][k] };
			t3 = _mm_load_ps(pack);
			//t3 = _mm_load1_ps(a[i] + k);
			j = k + 1;
			if (j % 4 != 0) {
				int count = (j / 4 + 1) * 4 < n ? (j / 4 + 1) * 4 : n;
				for (; j < count; j++) {
					a[i][j] = a[i][j] - a[i][k] * a[k][j];
				}
				/*for (int count = 4 - j % 4; count > 0 && j < n; count--, j++)
					a[i][j] = a[i][j] - a[i][k] * a[k][j];*/
			}
			for (; j + 3 < n; j += 4) {
				t1 = _mm_load_ps(a[i] + j);
				t4 = _mm_load_ps(a[k] + j);
				t4 = _mm_mul_ps(t3, t4);
				t1 = _mm_sub_ps(t1, t4);
				_mm_store_ps(a[i] + j, t1);
			}
			for (; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}

	}
}
void lu_avx(int n, float a[][maxN]) {
	__m256 t1, t2, t3, t4;
	for (int k = 0; k < n; k++) {
		float pack[8] = { a[k][k],a[k][k], a[k][k], a[k][k], a[k][k],a[k][k], a[k][k], a[k][k] };
		t2 = _mm256_loadu_ps(pack);
		//t2 = _mm_load1_ps(a[k] + k);
		int j = k + 1;
		for (; j + 7 < n; j += 8) {
			t1 = _mm256_loadu_ps(a[k] + j);
			t1 = _mm256_div_ps(t1, t2);
			_mm256_storeu_ps(a[k] + j, t1);
		}
		for (; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			float pack[4] = { a[i][k],a[i][k], a[i][k], a[i][k] };
			t3 = _mm256_loadu_ps(pack);
			//t3 = _mm_load1_ps(a[i] + k);
			j = k + 1;
			for (; j + 7 < n; j += 8) {
				t1 = _mm256_loadu_ps(a[i] + j);
				t4 = _mm256_loadu_ps(a[k] + j);
				t4 = _mm256_mul_ps(t3, t4);
				t1 = _mm256_sub_ps(t1, t4);
				_mm256_storeu_ps(a[i] + j, t1);
			}
			for (; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}

	}
}

void lu_sse_align2(int n, float a[][maxN]) {
	__m128 t1, t2, t3, t4;
	int temp = 0;
	for (int k = 0; k < n; k++) {
		float pack[4] = { a[k][k],a[k][k], a[k][k], a[k][k] };
		t2 = _mm_loadu_ps(pack);
		//t2 = _mm_load1_ps(a[k] + k);
		//int j = k + 1;
		temp = k + 1;
		int j = temp - temp % 4;
		for (; j + 3 < n; j += 4) {
			t1 = _mm_loadu_ps(a[k] + j);
			t1 = _mm_div_ps(t1, t2);
			_mm_storeu_ps(a[k] + j, t1);
		}
		for (; j < n; j++) {
			a[k][j] = a[k][j] / pack[0];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			float pack[4] = { a[i][k],a[i][k], a[i][k], a[i][k] };
			t3 = _mm_loadu_ps(pack);
			//t3 = _mm_load1_ps(a[i] + k);
			//j = k + 1;
			temp = k + 1;
			j = temp - temp % 4;
			for (; j + 3 < n; j += 4) {
				t1 = _mm_loadu_ps(a[i] + j);
				t4 = _mm_loadu_ps(a[k] + j);
				t4 = _mm_mul_ps(t3, t4);
				t1 = _mm_sub_ps(t1, t4);
				_mm_storeu_ps(a[i] + j, t1);
			}
			for (; j < n; j++) {
				a[i][j] = a[i][j] - pack[0] * a[k][j];
			}
			a[i][k] = 0;
		}

	}
}
void lu_sse_unfold(int n, float a[][maxN]) {
	__m128 t1, t2, t3, t4, t5;
	__m128 t6;
	for (int k = 0; k < n; k++) {
		float pack[4] = { a[k][k],a[k][k], a[k][k], a[k][k] };
		t2 = _mm_loadu_ps(pack);
		//t2 = _mm_load1_ps(a[k] + k);
		int j = k + 1;
		for (; j + 15 < n; j += 16) {
			t1 = _mm_loadu_ps(a[k] + j);
			t3 = _mm_loadu_ps(a[k] + j + 4);
			t4 = _mm_loadu_ps(a[k] + j + 8);
			t5 = _mm_loadu_ps(a[k] + j + 12);
			t1 = _mm_div_ps(t1, t2);
			t3 = _mm_div_ps(t3, t2);
			t4 = _mm_div_ps(t4, t2);
			t5 = _mm_div_ps(t5, t2);
			_mm_storeu_ps(a[k] + j, t1);
			_mm_storeu_ps(a[k] + j + 4, t3);
			_mm_storeu_ps(a[k] + j + 8, t4);
			_mm_storeu_ps(a[k] + j + 12, t5);
		}
		for (; j < n; j++) {
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;

		for (int i = k + 1; i < n; i++) {
			float pack[4] = { a[i][k],a[i][k], a[i][k], a[i][k] };
			t3 = _mm_loadu_ps(pack);
			//t3 = _mm_load1_ps(a[i] + k);
			j = k + 1;
			for (int count = 4 - j % 4; count > 0; count--, j++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j + 15 < n; j += 16) {
				t1 = _mm_loadu_ps(a[k] + j);
				t2 = _mm_loadu_ps(a[k] + j);
				t4 = _mm_loadu_ps(a[k] + j);
				t5 = _mm_loadu_ps(a[k] + j);

				t1 = _mm_mul_ps(t3, t1);
				t2 = _mm_mul_ps(t3, t2);
				t4 = _mm_mul_ps(t3, t4);
				t5 = _mm_mul_ps(t3, t5);

				t6 = _mm_loadu_ps(a[i] + j);
				t1 = _mm_sub_ps(t6, t1);
				t6 = _mm_loadu_ps(a[i] + j + 4);
				t2 = _mm_sub_ps(t6, t2);
				t6 = _mm_loadu_ps(a[i] + j + 8);
				t4 = _mm_sub_ps(t6, t4);
				t6 = _mm_loadu_ps(a[i] + j + 12);
				t5 = _mm_sub_ps(t6, t5); 
				
				_mm_storeu_ps(a[i] + j, t1);
				_mm_storeu_ps(a[i] + j + 4, t2);
				_mm_storeu_ps(a[i] + j + 8, t4);
				_mm_storeu_ps(a[i] + j + 12, t5);
			}
			for (; j < n; j++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}

	}
}
void check(int n, float a[][maxN]) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(10) << a[i][j];
			cout << " ";
		}
		cout << endl;
	}
	cout << endl;
}

int main() {

	ofstream  result1("lu_sse_.txt ", ios::app);
	ofstream  result2("lu_sse_align.txt ", ios::app);
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

	init(maxN, A);

	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu_sse(maxN, A);
	//check(maxN, A);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("lu_sse: %lfms.\n", (tail - head) * 1000.0 / freq);
	result1 << (tail - head) * 1000.0 / freq << endl;

	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	lu_sse_align(maxN, B);
	//check(maxN, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("lu_align: %lfms.\n", (tail - head) * 1000.0 / freq);
	result2 << (tail - head) * 1000.0 / freq << endl;

	result1.close();
	result2.close();



	//lu_sse_align(maxN, C);
	//check(maxN, C);
	//lu_avx(maxN, C);
	//check(maxN, C);
	//lu_sse_align2(maxN, C);
	//check(maxN, C);
	//lu_sse_unfold(maxN, A);
	//check(maxN, A);

}