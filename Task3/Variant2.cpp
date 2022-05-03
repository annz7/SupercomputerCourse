#include <iostream>
#include <mpi.h>
#include <cmath>

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

void UpdateX(double* x, double* newX, double* A, double* b, double tau, int numRows, int numCol, int startRow)
{
    for (int i = 0; i < numRows; i++)
    {
        double delta = 0;
        for (int j = 0; j < numCol; j++)
        {
            delta += A[i * numCol + j] * x[j];
        }

        newX[i] = x[i + startRow] - tau * (delta - b[i]);
    }
}

double GetErrorNuminator(double* A, double* x, double* b, int numRows, int numCol)
{
    double num = 0;
    for (int i = 0; i < numRows; i++)
    {
        double value = 0;
        for (int j = 0; j < numCol; j++)
        {
            value += A[i * numCol + j] * x[j];
        }

        value -= b[i];
        num += value * value;
    }

    return num;
}


int main(int argc, char* argv[])
{
    int processNum, rank;
    MPI_Init(&argc, &argv); // Инициализация MPI
    MPI_Comm_size(MPI_COMM_WORLD, &processNum); // Получение числа процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение номера процесса

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    const int size = 10;
    const double eps = 1E-8;
    const double tau = 0.0001;

    int rowsNum = size / processNum;
    if (size % processNum > rank)
        rowsNum++;

    double* xpart = new double[rowsNum];
    FillX(xpart, rowsNum);
    double* newXpart = new double[rowsNum];
    FillX(xpart, rowsNum);
    double* bpart = new double[rowsNum];
    FillB(bpart, rowsNum);

    double* finalX = new double[size];

    if (rank == 0) // основной процесс будет хранить всю инфу и общаться с другими
    {
        double* A = new double[size * size];
        FillA(A, size);

        double* Apart = new double[rowsNum * size];
        Apart = A;

        int rowId = rowsNum; // с какой строки начинаются значения текущего процесса
        int* rowsNums = new int[processNum]; // с каких строк начинаются значения процессов
        rowsNums[0] = rowsNum;
        for (int i = 1; i < processNum; i++)
        {
            rowsNums[i] = size / processNum;
            if (size % processNum > i)
                rowsNums[i]++;

            MPI_Ssend(A + rowId * size, rowsNums[i] * size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            MPI_Ssend(&rowId, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            rowId += rowsNums[i];
        }

        double error = 100;
        bool isCalc = true;

        while (error > eps)
        {
            double* x = new double[size];
            x = xpart;
            int idX = rowsNum;

            for (int i = 1; i < processNum; i++)
                MPI_Ssend(&isCalc, 1, MPI_C_BOOL, i, 3, MPI_COMM_WORLD); //даем отмашку считать новый Х

            for (int i = 1; i < processNum; i++) // получаем и склеиваем х
            {
                double* tempx = new double[rowsNums[i]];
                MPI_Recv(tempx, rowsNums[i], MPI_DOUBLE, i, 31, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < rowsNums[i]; j++)
                {
                    x[idX] = tempx[j];
                    idX++;
                }
            }

            for (int i = 1; i < processNum; i++)
                MPI_Ssend(x, size, MPI_DOUBLE, i, 32, MPI_COMM_WORLD);

            UpdateX(x, newXpart, Apart, bpart, tau, rowsNum, size, 0);

            for (int i = 0; i < rowsNum; i++)
                xpart[i] = newXpart[i];

            idX = rowsNum;
            x = xpart;
            for (int i = 1; i < processNum; i++) // получаем части новых Х и склеиваем
            {
                double* x_temp = new double[rowsNums[i]];
                MPI_Recv(x_temp, rowsNums[i], MPI_DOUBLE, i, 35, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int j = 0; j < rowsNums[i]; j++)
                {
                    x[idX] = x_temp[j];
                    idX++;
                }
            }


            for (int i = 1; i < processNum; i++)
                MPI_Ssend(x, size, MPI_DOUBLE, i, 4, MPI_COMM_WORLD); // отправляем итоговый Х на подсчет ошибок

            double errorNum = GetErrorNuminator(Apart, x, bpart, rowsNum, size);

            double errorDenom = 0;

            for (int i = 1; i < processNum; i++)
            {
                double currErrorNum;
                MPI_Recv(&currErrorNum, 1, MPI_DOUBLE, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                errorNum += currErrorNum;
            }

            //std::cout << "error " << errorNum << std::endl;


            for (int i = 0; i < rowsNum; i++)
                errorDenom += bpart[i] * bpart[i];

            for (int i = 1; i < processNum; i++)
            {
                double* b_temp = new double[rowsNums[i]];
                MPI_Recv(b_temp, rowsNums[i], MPI_DOUBLE, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < rowsNums[i]; j++)
                    errorDenom += b_temp[j] * b_temp[j];
            }

            error = sqrt(errorNum / errorDenom);

            for (int i = 0; i < size; i++)
                finalX[i] = x[i];
        }

        for (int i = 0; i < size; i++)
            std::cout << finalX[i] << ' ';

        isCalc = false; // завершаем процессы
        for (int i = 1; i < processNum; i++)
            MPI_Ssend(&isCalc, 1, MPI_C_BOOL, i, 3, MPI_COMM_WORLD);

        double finish = MPI_Wtime();
        std::cout << "\nTime: " << finish - start << std::endl;
    }
    else
    {
        double* Apart = new double[rowsNum * size];
        int rowId;
        bool isCalc;

        MPI_Recv(Apart, rowsNum * size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //получаем матрицу
        MPI_Recv(&rowId, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //получаем номер строки начала матрицы

        while (true)
        {
            MPI_Recv(&isCalc, 1, MPI_C_BOOL, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // получаем отмашку считать

            if (!isCalc)
                break;

            MPI_Ssend(xpart, rowsNum, MPI_DOUBLE, 0, 31, MPI_COMM_WORLD);

            //обновляем часть newX и отправляем в основной
            double* x = new double[size];
            MPI_Recv(x, size, MPI_DOUBLE, 0, 32, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            UpdateX(x, newXpart, Apart, bpart, tau, rowsNum, size, rowId);

            for (int i = 0; i < rowsNum; i++)
                xpart[i] = newXpart[i];

            MPI_Ssend(xpart, rowsNum, MPI_DOUBLE, 0, 35, MPI_COMM_WORLD);

            //считаем локальную ошибку на новом Х
            MPI_Recv(x, size, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /*for (int i = 0; i < size; i++)
                std::cout << x[i] << ' ';
            std::cout << std::endl;*/

            double errorNum = GetErrorNuminator(Apart, x, bpart, rowsNum, size);
            MPI_Ssend(&errorNum, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
            MPI_Ssend(bpart, rowsNum, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
        }

    }

    MPI_Finalize(); // Завершение работы MPI

    return 0;
}