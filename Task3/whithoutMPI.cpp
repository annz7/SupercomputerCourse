#include <cstdio>
#include <cmath>
#include <iostream>
#include "time.h"

void FillX(double* x, int size)
{
    for (int i = 0; i < size; i++)
        x[i] = 0;
}

void FillB(double* b, int size)
{
    for (int i = 0; i < size; i++)
        b[i] = size + 1;
}

void FillA(double* A, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j)
                A[i * size + j] = 2;
            else
                A[i * size + j] = 1;
        }
    }
}

void UpdateX(double* x, double* newX, double* A, double* b, double tau, int size)
{
    for (int i = 0; i < size; i++)
    {
        double delta = 0;
        for (int j = 0; j < size; j++)
        {
            delta += A[i * size + j] * x[j];
        }

        newX[i] = x[i] - tau * (delta - b[i]);
    }
}

double GetError(double* A, double* x, double* b, int size)
{
    double nom = 0;
    double denom = 0;
    for (int i = 0; i < size; i++)
    {
        denom += b[i] * b[i];
        
        double value = 0;
        for (int j = 0; j < size; j++)
        {
            value += A[i * size + j] * x[j];
        }

        value -= b[i];
        nom += value * value;
    }

    return sqrt(nom) / sqrt(denom);
}


int main(int argc, char* argv[]) 
{
    clock_t start = clock();

    const int size = 3;
    const double eps = 1E-10;
    const double tau = 0.00001;

    double* x = new double[size];
    double* newX = new double[size];
    double* b = new double[size];
    double* A = new double[size * size];
    FillX(x, size);
    FillX(newX, size);
    FillB(b, size);
    FillA(A, size);

    double error = 10;

    while (error > eps)
    {
        UpdateX(x, newX, A, b, tau, size);

        error = GetError(A, newX, b, size);

        for (int i = 0; i < size; i++)
            x[i] = newX[i];
    }

    clock_t finish = clock();

    for (int i = 0; i < size; i++)
        std::cout << x[i] << ' ';

    std::cout << "\nTime: " << (double)(finish - start) / CLOCKS_PER_SEC;

    return 0;
}