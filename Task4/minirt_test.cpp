#include "minirt/minirt.h"

#include <cmath>
#include <iostream>

#include <thread>
#include <vector>
#include <queue>
#include <chrono>
#include <mutex>
#include <condition_variable>

using namespace minirt;
using namespace std;

void initScene(Scene& scene) {
    Color red{ 1, 0.2, 0.2 };
    Color blue{ 0.2, 0.2, 1 };
    Color green{ 0.2, 1, 0.2 };
    Color white{ 0.8, 0.8, 0.8 };
    Color yellow{ 1, 1, 0.2 };

    Material metallicRed{ red, white, 50 };
    Material mirrorBlack{ Color {0.0}, Color {0.9}, 1000 };
    Material matteWhite{ Color {0.7}, Color {0.3}, 1 };
    Material metallicYellow{ yellow, white, 250 };
    Material greenishGreen{ green, 0.5, 0.5 };

    Material transparentGreen{ green, 0.8, 0.2 };
    transparentGreen.makeTransparent(1.0, 1.03);
    Material transparentBlue{ blue, 0.4, 0.6 };
    transparentBlue.makeTransparent(0.9, 0.7);

    scene.addSphere(Sphere{ {0, -2, 7}, 1, transparentBlue });
    scene.addSphere(Sphere{ {-3, 2, 11}, 2, metallicRed });
    scene.addSphere(Sphere{ {0, 2, 8}, 1, mirrorBlack });
    scene.addSphere(Sphere{ {1.5, -0.5, 7}, 1, transparentGreen });
    scene.addSphere(Sphere{ {-2, -1, 6}, 0.7, metallicYellow });
    scene.addSphere(Sphere{ {2.2, 0.5, 9}, 1.2, matteWhite });
    scene.addSphere(Sphere{ {4, -1, 10}, 0.7, metallicRed });

    scene.addLight(PointLight{ {-15, 0, -15}, white });
    scene.addLight(PointLight{ {1, 1, 0}, blue });
    scene.addLight(PointLight{ {0, -10, 6}, red });

    scene.setBackground({ 0.05, 0.05, 0.08 });
    scene.setAmbient({ 0.1, 0.1, 0.1 });
    scene.setRecursionLimit(20);

    scene.setCamera(Camera{ {0, 0, -20}, {0, 0, 0} });
}

struct Point
{
    int x;
    int y;

    Point(int x, int y) : x(x), y(y) {}
};

struct Task
{
    vector<Point> points;
    Task(vector<Point> points) : points(points) {}
};

template<typename T>
class Queue
{
public:
    queue<T> queue_;
    mutex locker;
    condition_variable isNotEmpty;

    void Push(T value)
    {
        lock_guard<mutex> guard(locker);
        queue_.push(value);
        isNotEmpty.notify_one();
    }

    T Pop()
    {
        unique_lock<mutex> lock(locker);
        while (queue_.empty())
            isNotEmpty.wait(lock);
        return PopLocked();
    }

    T PopLocked()
    {
        T front = queue_.front();
        queue_.pop();
        return front;
    }
};


class ThreadsPool
{
public:
    Queue<Task> tasks;
    vector<thread> threads;
    Scene scene;
    Image image;
    ViewPlane viewPlane;
    int threadsNum;
    int numOfSamples;

    ThreadsPool(int threadsNum, const Scene& scene, const Image& image, ViewPlane& viewPlane, int numOfSamples): 
        viewPlane(viewPlane)
    {
        this->threadsNum = threadsNum;
        this->scene = scene;
        this->image = image;
        this->numOfSamples = numOfSamples;
        InitThreads(threadsNum);
    };

    void InitThreads(int threadsNum)
    {
        for (int i = 0; i < threadsNum; i++)
            threads.emplace_back([this]() { ThreadFunction(); });
    }

    void ThreadFunction()
    {
        auto task = tasks.Pop();
        while (task.points[0].x >= 0) // point(-1, -1) - stop element
        {
            CalculateTask(task);
            task = tasks.Pop();
        }
    }

    void CalculateTask(Task task)
    {
        for (int i = 0; i < task.points.size(); i++)
        {
            CalculatePoint(task.points[i]);
        }
    }

    void CalculatePoint(Point point)
    {
        const auto color = viewPlane.computePixel(scene, point.x, point.y, numOfSamples);
        image.set(point.x, point.y, color);
    }

    void AddTaskToQueue(Task task)
    {
        tasks.Push(task);
    }

    void Join()
    {
        for (int i = 0; i < threadsNum; i++)
        {
            vector<Point> endTask;
            endTask.push_back({ -1,-1 });
            tasks.Push(endTask); // add stop element to queue for every thread
        }

        for (int i = 0; i < threadsNum; i++)
            threads[i].join();

        threads.clear();
    }
};

int main(int argc, char** argv) {
    int viewPlaneResolutionX = (argc > 1 ? stoi(argv[1]) : 600);
    int viewPlaneResolutionY = (argc > 2 ? stoi(argv[2]) : 600);
    int numOfSamples = (argc > 3 ? stoi(argv[3]) : 1);
    string sceneFile = (argc > 4 ? argv[4] : "");

    Scene scene;
    if (sceneFile.empty()) {
        initScene(scene);
    }
    else {
        scene.loadFromFile(sceneFile);
    }

    const double backgroundSizeX = 4;
    const double backgroundSizeY = 4;
    const double backgroundDistance = 15;

    const double viewPlaneDistance = 5;
    const double viewPlaneSizeX = backgroundSizeX * viewPlaneDistance / backgroundDistance;
    const double viewPlaneSizeY = backgroundSizeY * viewPlaneDistance / backgroundDistance;

    ViewPlane viewPlane{ viewPlaneResolutionX, viewPlaneResolutionY,
                         viewPlaneSizeX, viewPlaneSizeY, viewPlaneDistance };

    Image image(viewPlaneResolutionX, viewPlaneResolutionY); // computed image

    auto start = chrono::high_resolution_clock::now();

    auto threadsPool = new ThreadsPool(3, scene, image, viewPlane, numOfSamples);

    for (int x = 0; x < viewPlaneResolutionX; x++)
    {
        vector<Point> task;
        for (int y = 0; y < viewPlaneResolutionY; y++)
            task.push_back({ x, y });
        threadsPool->AddTaskToQueue(task);
    }

    threadsPool->Join();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time = end - start;
    cout << "time " << time.count() << endl;

    image.saveJPEG("raytracing.jpg");

    return 0;
}
