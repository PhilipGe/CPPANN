// thread example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <pthread.h>         // std::thread

using namespace std;

int count = 0;

void foo() 
{
  std::cout<<"Foo "<<count<<std::endl;

  for(int i = 0;i < 1000;i++){
    count ++;
  }
}

void bar(int x)
{
  // do stuff...
  cout<<"Bar "<<count<<" num: "<<x<<endl;
}

int main() 
{
  std::thread first (foo);     // spawn new thread that calls foo()
  std::thread second (bar,0);  // spawn new thread that calls bar(0)

  std::cout << "main, foo and bar now execute concurrently...\n";

  // synchronize threads:
  first.join();                // pauses until first finishes
  second.join();               // pauses until second finishes

  std::cout << "foo and bar completed.\n";
  std::cout << "count: "<<count<<endl;

  return 0;
}