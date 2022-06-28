
// Versione con ricerca sulle mappe (solo per mstati_di_mat, non per statim) e PARALLELIZZAZIONE BUONA

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <ctime>
#include <map>
//#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <future>
#include <mutex>
#include <assert.h>

#include <cmath>
#include <chrono>
#include <random>
extern "C"
{    
    #include <cblas.h> 
}
//#include <oneapi/mkl.hpp>

#define HASH_MULT  314159    /* random multiplier */
#define HASH_PRIME 516595003 /* the 27182818th prime; it's $\leq 2^{29}$ */
#define N 12//14//5//14


using namespace std;

namespace py = pybind11;


using mappa = map<int, unsigned long long int>;

int show = 0;

// Definisco le funzioni dichiarate


// # Funzione per creare la matrice
void matrice(double* mat, double eps, double dil) { 
    double s_mat[N*N];
    double a_mat[N*N];
    unsigned long seed = 9999;//chrono::system_clock::now().time_since_epoch().count();
    default_random_engine re (seed);
    uniform_real_distribution<double> unif(0,1);
    double a1, a2; 

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i+j*N] = 0;
            a_mat[i+j*N] = 0;
            s_mat[i+j*N] = 0;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (unif(re) > dil){
              a1 = (unif(re) - 0.5) * 2;                         // Gli elementi sono diversi da 0 solo con prob (1-dil)
              s_mat[i+j*N] = a1;                                  // Creo le due matrici, una simmetrica e una asimmetrica
              s_mat[j+i*N] = a1;                      
            }
            if (unif(re) > dil) {
              a2 = (unif(re) - 0.5) * 2;
              a_mat[i+j*N] = a2;
              a_mat[j+i*N] = -a2;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            mat[i+j*N] = (1 - eps / 2) * s_mat[i+j*N] + (eps / 2) * a_mat[i+j*N];   // La diagonale viene nulla per costruzione
        }
    }

}

// # Calcolo il prodotto tra matrice e vettore
void dot(double* mat, int* in, double* out) {
  // print net
  //py::print("net in dot");
  //for (unsigned int j=0; j<N;++j){
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(mat[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
	//}
  //py::print("N",N);
  double o;            
  for (int j = 0; j < N; ++j) {
    //py::list line;
    o=0; 
    for (int i = 0; i < N; ++i) {
        o += mat[i+N*j] * (double) in[i];
        //line.append("+");
        //line.append(mat[i+j*N]);
        //line.append("*");
        //line.append(in[i]);
    }
    //line.append(o);
    out[j]=o;
    //py::print("j",j,o,"=",line);
  }
}


// # Creo lo stato "n"
void s_da_n(int* s0, int n) {
    for (int i = 0; i < N; i++) {
        s0[N-i-1] = (int(double(n) / (1 << i))) % 2;           // 1<<i è 2**i, calcolato molto velocemente
    }
}

// # Calcolo "n" corrispondente allo stato "s"
int n_da_s(int* s) {
    int ret = 0;

    for (int i = 0; i < N; i++){
        ret += s[i] * (1 << (N - 1 - i));               // Ovvero +=s[i]*2**(N-1-i)
    }

    return ret;
}


// # Calcolo della media da una mappa
double mean_dic(const map<int, unsigned long long int> &a) {
    double sum = 0, dot = 0;
    for (map<int, unsigned long long int>::const_iterator it = a.cbegin(); it != a.cend(); ++it) {
        sum += it->second;
        dot += it->first * it->second;
    }

    return dot / sum;
}

// # Calcolo della deviazione standard da una mappa
double std_dic(const map<int, unsigned long long int> &a) {
    double sum = 0, dot = 0, dotsq = 0;
    for (map<int, unsigned long long int>::const_iterator it = a.cbegin(); it != a.cend(); ++it) {
        sum += it->second;
        dot += it->first * it->second;
        dotsq += (it->first) * (it->first) * it->second;
    }

    dot /= sum;
    dotsq /= sum;

    return sqrt(dotsq - (dot * dot));
}

void ciclo_singola_matrice(double eps,
                           double dil,
                           double thresh,
                           mappa *isto_dist,
                           mappa *isto_lung,
                           mappa *isto_size,
                           mappa *isto_nclu,
                           mappa *isto_nvic,
                           int ripetiz,
                           bool loadNets,
                           double* nets) {
  double mat[N*N];
  for (int rip = 0; rip < ripetiz; ++rip) {
        //py::print("loadNet",loadNets);
        if(loadNets == false){
          matrice(mat,eps, dil);     // Matrice delle connessioni
        }else{
          for (unsigned int j = 0; j<N;++j){
            for (unsigned int i= 0; i<N;++i){
              mat[i+j*N] = nets[i+j*N+rip*N*N];
            }    
          }
        }

        if (show==1){
          for (unsigned int j=0; j<N;++j){
            py::list line;
            for (unsigned int i = 0; i < N; ++i) {
              line.append(mat[i+j*N]);
            }
            py::print(j,line);
          }
	      }

        int icluster = 0;                      // Contatore dei cluster
        //unordered_map<vector<int>, int, VectHash<int>> statiVisitati{};                 // Lista degli visitati
        const unsigned int numStati = (1 << N);
        map<int, int> statiVisitati;          // Lista degli visitati
        map<int, int> path;          // Lista degli visitati
        int charact[numStati];                // Carattere degli stati esplorati (1 inizio, 2 transiente, 3 ciclo limite)
        py::list allCycles;
        int whichCycle[numStati];                // C.L. degli stati
        int dist[numStati];                   // Distanza dal C.L. degli stati
        int nvic[numStati];                   // Numero stati che evolvono in un certo stato
        int si[N];                            // stati iniziale espresso come vettore binario
        //unsigned int sf[N];                            // stati finale espresso come vettore binario
        double v[N];                            // somma pesata
        unsigned int k0;
        unsigned int k;                       
        unsigned int t;
    
        for (unsigned int n = 0; n < numStati; ++n) {                      // Ciclo sui 2^N stati iniziali
           py::list cycle;
           path.clear();
           k0 = n;                           // Lista degli stati singoli di una condiz iniziale, in base 10
           if (show==1) py::print("k0",k0);
           
           if (statiVisitati.find(k0) != statiVisitati.end()) continue;  //  Controllo che uno stato come quello iniziale (ovvero [0,0...0, n]) non sia già comparso
           path[k0] = 1;
           statiVisitati[k0] = 1;      // Aggiungo alla mappa il nuovo vettore e l'indice corrispondente (ovvero la lunghezza della mappa)
           charact[k0]=1;
           nvic[k0]=0;

           t = 0;                                                     // Contatore del tempo del processo
           s_da_n(si,n);
           if (show==1){                                   // Stato come vettore di attivazione dei neuroni
            py::list vec;
            for (unsigned int i=0; i<N;++i) vec.append(si[i]);
            py::print("vec",vec);
           }


           while (true) {
               t += 1;   // Lo aggiorno subito perchè nello 0 dele liste già ho messo lo stato iniziale (quindi voglio che alla prima iterazione valga 1)
               dot(mat, si, v);
               //if (show==1){              // Stato come vettore di attivazione dei neuroni
               //   py::list vec;
               //   for (unsigned int i=0; i<N;++i) vec.append(v[i]);
               //   py::print("weighted sum",vec);
               //}
               // Heavyside for each element in v 
               for (int i = 0; i < N; ++i) {
                   si[i] = (v[i] >= thresh);//>= thresh);  for backwords comparrison 
               }  

               k = n_da_s(si);          // Ricavo il valore decimale corrispondente allo stato
               if (show==1){  
                py::print(">k",k);                                 // Stato come vettore di attivazione dei neuroni
                py::list vec;
                for (unsigned int i=0; i<N;++i) vec.append(si[i]);
                py::print("vec",vec);
               }

               if (statiVisitati.find(k) == statiVisitati.end()) {
                  nvic[k]=1;
                  charact[k]=2;
                  statiVisitati[k]=1; //Aggiungi agli stati visitati
                  path[k]=1;
                  if (show==1){
                    py::list vec;
                    for(std::map<int,int>::iterator iter = statiVisitati.begin(); iter != statiVisitati.end(); ++iter){
                      int key =  iter->first;
                      vec.append(key);
                    }
                    py::print("statiVisitati",vec);
                  }
               } else {
                  std::map<int,int>::iterator iterPath = path.find(k);
                  if (iterPath != path.end()) {
                     icluster += 1;
                     nvic[k] += 1;
                     //all states in cycle must be assigned character 3, that says they are cycles
                     for(std::map<int,int>::iterator iterCycle = iterPath; iterCycle != path.end(); ++iterCycle){
                       int w = iterCycle->first;
                       charact[w]=3;
                       cycle.append(w);
                     } 
                     allCycles.append(cycle);
                     if (show==1){
                       py::print("cycle",cycle);
                       py::print("found cycles",allCycles);
                     }
                     //all states in path must be assigned to the corresponding cycle
                     for(std::map<int,int>::iterator iterPath2 = path.begin(); iterPath2 != path.end(); ++iterPath2){
                       whichCycle[iterPath2->first]=icluster;
                     } 
                     break;
                   }else{
                     nvic[k] += 1; 
                     int c = whichCycle[k];
                     for(std::map<int,int>::iterator iterPath2 = path.begin(); iterPath2 != path.end(); ++iterPath2){
                       if((int) k != iterPath2->first) whichCycle[iterPath2->first]=c;
                     } 
                     break;                
                   }
               }
           }

        }
        
        if (show==1){  
          py::list vec;
          for (unsigned int i=0; i<numStati;++i) vec.append(whichCycle[i]);
          py::print("vec",vec);
        }

        //mtx.lock();
        for (int i = 0; i < (1<<N); ++i) {                                  
            (*isto_nvic)[nvic[i]] += 1;                                        
//            (*isto_dist)[dist[i]] += 1;                                        
//            if (charact[i] == 1) (*isto_dmax)[dist[i]] += 1;                   
        }

        for (int i = 1; i < icluster+1; ++i) {
            int lunghezza = 0, bacino = 0;
            for (unsigned int j = 0; j < numStati; ++j) {
                lunghezza += ((whichCycle[j]==i) && (charact[j]==3));       // se lo stato j appartiene al cluster e ha carattere 3, è cl
                bacino += (whichCycle[j] == i);
            }
            (*isto_lung)[lunghezza] += 1;
            (*isto_size)[bacino] += 1;
        }

        (*isto_nclu)[icluster] += 1;
        //mtx.unlock();
  }
}


py::dict runRHNN(double eps, double dil,int ripetiz = 1000){
  mappa isto_dist;                                  // Rappresento gli istogrammi come mappe
  //mappa isto_dmax;
  mappa isto_lung;
  mappa isto_size;
  mappa isto_nclu;
  mappa isto_nvic;

  py::dict dict_measures;

  double *ptrNet;
  //ptrNet[0] = 0;
  //double eps = 0.0;
  //double dil = 0.0;
  py::print("N",N,"eps",eps, "dil",dil,"ripetiz",ripetiz);
  //int m = 1;
  //int refr = 0; 
  double thresh = 0.;
  //double tau = 1.0;
  //vector<double> f_k = generate_f_k(m, tau);        // Vettore dei decadimenti
  //int ripetiz = 1;

  ciclo_singola_matrice(eps, dil, thresh,
                        &isto_dist, &isto_lung, &isto_size, &isto_nclu, &isto_nvic,
                        ripetiz, false, ptrNet);

  // Calcola medie
  double meanDist = mean_dic(isto_dist);
  double meanSize = mean_dic(isto_size);
  double meanLung = mean_dic(isto_lung);
  double meanNClu = mean_dic(isto_nclu);
  //mettile nel dizionario
  dict_measures["Dist"] = meanDist;
  dict_measures["Size"] = meanSize;
  dict_measures["Lung"] = meanLung;
  dict_measures["NClu"] = meanNClu;

  return dict_measures;
}

py::dict runRHNNwithNets(double thresh, py::array_t<double> net){
  
  py::buffer_info bufNet = net.request();
  double *ptrNet = (double *) bufNet.ptr;
  unsigned int Np = (unsigned int) bufNet.shape[2];
  unsigned int Npp = (unsigned int)  bufNet.shape[1];
  unsigned int ripetiz = (unsigned int) bufNet.shape[0];
  py::print("N",N);
  py::print("Np",Np);
  py::print("Npp",Npp);
  assert (N==Np);  // change line 11 # define N 14
  assert (N==Npp);
  py::print("ripetiz",ripetiz);

  //for (unsigned int k=0; k<ripetiz;++k){
  //  for (unsigned int j=0; j<N;++j){
  //    py::list line;
  //    for (unsigned int i = 0; i < N; ++i) {
  //      line.append(ptrNet[i+j*N+k*N*N]);
  //      //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //    }
  //    py::print(j,k,line);
  //  }
	//}
  
  mappa isto_dist;                                  // Rappresento gli istogrammi come mappe
  //mappa isto_dmax;
  mappa isto_lung;
  mappa isto_size;
  mappa isto_nclu;
  mappa isto_nvic;

  py::dict dict_measures;


  //double eps = 0.0;
  //double dil = 0.0;
  py::print("N",N,"thresh",thresh,"ripetiz",ripetiz);
  //int m = 1;
  //int refr = 0; 
  double threshold = 0.;
  //double tau = 1.0;
  //vector<double> f_k = generate_f_k(m, tau);        // Vettore dei decadimenti
  //int ripetiz = 1;

  ciclo_singola_matrice(0.0, 0.0, threshold,
                        &isto_dist, &isto_lung, &isto_size, &isto_nclu, &isto_nvic,
                        ripetiz,true,ptrNet);

  // Calcola medie
  double meanDist = mean_dic(isto_dist);
  double meanSize = mean_dic(isto_size);
  double meanLung = mean_dic(isto_lung);
  double meanNClu = mean_dic(isto_nclu);
  //mettile nel dizionario
  dict_measures["Dist"] = meanDist;
  dict_measures["Size"] = meanSize;
  dict_measures["Lung"] = meanLung;
  dict_measures["NClu"] = meanNClu;

  return dict_measures;
}

void stateIndex2stateVecCpp(int input,  int* out, std::size_t size){
    for (unsigned int i = 0; i < size; i++) {
        out[size-i-1] = (int(double(input) / (1 << i))) % 2;           // 1<<i è 2**i, calcolato molto velocemente
    }
}

py::array_t<int> stateIndex2stateVec(int input, std::size_t size){
  py::array_t<int> out = py::array_t<int>(size);
	py::buffer_info bufout = out.request();
  int *ptrOut = (int *) bufout.ptr;
  stateIndex2stateVecCpp(input,  ptrOut, size);
	return out;
}


// # Calcolo "n" corrispondente allo stato "s"
int stateVec2stateIndexCpp(int* input, std::size_t size ) {
    int ret = 0;
    for (unsigned int i = 0; i < size; i++){
        ret += input[i] * (1 << (size - 1 - i));               // Ovvero +=s[i]*2**(N-1-i)
    }
    return ret;
}

int stateVec2stateIndex(py::array_t<int> input){
  py::buffer_info bufinput = input.request();
  int *ptrInt = (int *) bufinput.ptr;
  unsigned int size = (unsigned int) bufinput.shape[0];
  int out;
  out=stateVec2stateIndexCpp(ptrInt, size);
	return out;
}

//    transiton function. net1 is the network that generates the ttransitions
//    Sigma_path0 is a binary vector it generates the corresponding transtions.     
//    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
//    typ=1 --> {-1,1}    typ=0 --> {0,1} 
void trans(int* sigma_path0,int* sigma_path1,double* net,unsigned int size,int typ = 1,int thr = 0,int signFuncInZero = 1) {                           // stato finale espresso come vettore binario
  double v[size];
  int temp;
  assert(N==size);
  //// print net
  //py::print("net in trans");
  //for (unsigned int j=0; j<N;++j){
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(net[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
	//}
  dot(net, sigma_path0, v);
  //sigma_path1 = net1.dot(sigma_path0.T)
  // Heavyside for each element in v 
  for (unsigned int i = 0; i < size; ++i) {
    temp = (v[i] > thr);//>= thr);  for backwords comparrison
    //py::print("i",i,v[i],temp );
    temp += (temp==0)*(v[i] == thr)*signFuncInZero;  //
    //py::print("i",i,v[i],temp );
    sigma_path1[i]=(1-typ + temp)/(2-typ);
  }  
  //sigma_path1 = (1-typ + sign(sigma_path1 +thr,signFuncInZero) )/(2-typ)
}

py::array_t<int> tranCpp(py::array_t<int> sigma_path0,py::array_t<double> net,int typ = 1,double thr = 0,int signFuncInZero = 1){
  //sigma_path1
  py::array_t<int> sigma_path1 = py::array_t<int>(N);
	py::buffer_info bufS1 = sigma_path1.request();
  int *ptrS1 = (int *) bufS1.ptr;
  //sigma_path0
  py::buffer_info bufS0 = sigma_path0.request();
  int *ptrS0 = (int *) bufS0.ptr;
  unsigned int size = (unsigned int) bufS0.shape[0];
  assert (N == size);
  //net
  py::buffer_info bufNet = net.request();
  double *ptrNet = (double *) bufNet.ptr;
  unsigned int Np = (unsigned int) bufNet.shape[1];
  unsigned int Npp = (unsigned int)  bufNet.shape[0];
  //py::print("N",N);
  //py::print("Np",Np);
  //py::print("Npp",Npp);
  assert (N == Np);  // change line 11 # define N 14
  assert (N == Npp);

  //// print net
  //for (unsigned int j=0; j<N;++j){
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrNet[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
	//}

  trans(ptrS0, ptrS1, ptrNet, Np, typ, thr, signFuncInZero);
  
  return  sigma_path1;
}

//    transiton function. net1 is the network that generates the ttransitions
//    sigma_path0 is a array of binary vectors it generates a array with the corresponding transtions.    
//    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
//    typ=1 --> {-1,1}    typ=0 --> {0,1} 
void trans(int* sigma_path0,int* sigma_path1,double* net, unsigned int numS0 ,unsigned int size,int typ = 1,int thr = 0,int signFuncInZero = 1) {  
  unsigned int k,j,i;
  double v[numS0*N];
  int temp;
  assert(N==size);

  // print net
  //py::print("net in dot");
  //for (unsigned int j=0; j<N;++j){
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(mat[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
	//}
  //py::print("N",N);
  //printf("S0 Cpp\n");
  //for (i = 0; i<3; ++i)//numS0
  //{
  //  for (j = 0; j<N; ++j)
  //  printf("%2.2f ", (double) sigma_path0[i * N + j]);
  //  putchar('\n');
  //}
  
  // compute activation
  double o;

  for (k = 0; k < numS0; ++k) {
    //py::print("k",k);
    for (j = 0; j < N; ++j) {
      //py::list line;
      o=0; 
      for (i = 0; i < N; ++i) {
	      o += net[i+N*j] * (double) sigma_path0[i+N*k];
	      //line.append("+");
	      //line.append(net[i+j*N]);
	      //line.append("*");
	      //line.append(sigma_path0[i]);
      }
      v[j+N*k]=o;
      //py::print("j",j,o,"=",line);
    }
  }

  //printf("v Cpp\n");
  //for (i = 0; i<3; ++i)//numS0
  //{
  //  for (j = 0; j<N; ++j)
  //  printf("%2.2f ", v[i * N + j]);
  //  putchar('\n');
  //}

  //sigma_path1 = net1.dot(sigma_path0.T)
  // Heavyside for each element in v 
  for (i = 0; i < numS0*N; ++i) {
    temp = (v[i] > thr);//>= thr);  for backwords comparrison
    //py::print("i",i,v[i],temp );
    temp += (temp==0)*(v[i] == thr)*signFuncInZero;  //
    //py::print("i",i,v[i],temp );
    sigma_path1[i]=(1-typ + temp)/(2-typ);
  }  
  //sigma_path1 = (1-typ + sign(sigma_path1 +thr,signFuncInZero) )/(2-typ)
} 

py::array_t<int> transManyStatesCpp(py::array_t<int> sigma_path0,py::array_t<double> net,int typ = 1,double thr = 0,int signFuncInZero = 1){
  //sigma_path0
  py::buffer_info bufS0 = sigma_path0.request();
  int *ptrS0 = (int *) bufS0.ptr;
  size_t numS0 = (unsigned int) bufS0.shape[0];
  size_t size = (unsigned int) bufS0.shape[1];
  assert (N == size);
  //sigma_path1
  py::array_t<int> sigma_path1 = py::array_t<int>(bufS0.shape);
	py::buffer_info bufS1 = sigma_path1.request();
  int *ptrS1 = (int *) bufS1.ptr;
  //net
  py::buffer_info bufNet = net.request();
  double *ptrNet = (double *) bufNet.ptr;
  unsigned int Np = (unsigned int) bufNet.shape[1];
  unsigned int Npp = (unsigned int)  bufNet.shape[0];
  //py::print("N",N);
  //py::print("Np",Np);
  //py::print("Npp",Npp);
  assert (N == Np);  // change line 11 # define N 14
  assert (N == Npp);

  //// print net
  //for (unsigned int j=0; j<N;++j){
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrNet[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
	//}

  //py::print("ptrS0\n");
  //for (unsigned int j=0; j<3;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrS0[i+j*N]);
  //  }
  //py::print(j,line);
  //}

  trans(ptrS0, ptrS1, ptrNet, numS0, N, typ, thr, signFuncInZero);
  
  //py::print("ptrS1\n");
  //for (unsigned int j=0; j<3;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrS1[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
  //}
  
  return  sigma_path1;
}


//    transiton function. net1 is the network that generates the ttransitions
//    sigma_path0 is a array of binary vectors it generates a array with the corresponding transtions.    
//    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
//    typ=1 --> {-1,1}    typ=0 --> {0,1} 
void transCBLAS(double* sigma_path0,double* sigma_path1,double* net, unsigned int numS0 ,unsigned int size,int typ = 1,double thr = 0,int signFuncInZero = 1) {  
  //double Vt[numS0*N];
  //py::print("pars", numS0, N,typ, thr, signFuncInZero);
  double V[numS0*N];
  int temp;
  assert(N==size);
  unsigned int i;
  
  int l=N; //number of rows in C, number of rows in A
  int n=numS0; //number of columns in C, number of row in B
  int m=N; //number of columns in A, number of columns in B
  //double V[numS0*N];
  //double S0[numS0*N];
  //for (j = 0; j<numS0*N; ++j) S0[j] = (double) sigma_path0[j];
  //printf("Check Trans CBlas\n");
  //printf("S0 CBlas\n");
  //for (i = 0; i<3; ++i)//numS0
  //{
  //  for (j = 0; j<N; ++j)
  //  printf("%2.2f ", S0[i * N + j]);
  //  putchar('\n');
  //}

  //printf("net CBlas\n");
  //for (i = 0; i<N; ++i)//numS0
  //{
  //  for (j = 0; j<N; ++j)
  //  printf("%2.2f ", net[i * N + j]);
  //  putchar('\n');
  //}
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, l, m, 1.0, sigma_path0, m, net, m,  0.0, V, l);

  //for (i = 0; i<N; ++i)//numS0
  //{
  //  for (j = 0; j<numS0; ++j)
  //  V[i * N + j] = Vt[i + j * numS0];
  //}

  //printf("V CBlas\n");
  //for (i = 0; i<3; ++i)//numS0
  //{
  //  for (j = 0; j<N; ++j)
  //  printf("%2.2f ", V[i * N + j]);
  //  putchar('\n');
  //}

  //sigma_path1 = net1.dot(sigma_path0.T)
  // Heavyside for each element in v 
  for (i = 0; i < numS0*N; ++i) {
    temp = (V[i] > thr);//>= thr);  for backwords comparrison
    //py::print("i",i,v[i],temp );
    temp += (temp==0)*(V[i] == thr)*signFuncInZero;  //
    //py::print("i",i,v[i],temp );
    sigma_path1[i]=(1-typ + temp)/(2-typ);
  }  
  //sigma_path1 = (1-typ + sign(sigma_path1 +thr,signFuncInZero) )/(2-typ)
  //printf("S1 CBlas %f\n",thr);
  //for (i = 0; i<3; ++i)//numS0
  //{
  //  for (j = 0; j<N; ++j)
  //  printf("%d ", sigma_path1[i * N + j]);
  //  putchar('\n');
  //}
} 

py::array_t<int> transManyStatesCBlas(py::array_t<double> sigma_path0,py::array_t<double> net,int typ = 1,double thr = 0,int signFuncInZero = 1){
  //sigma_path0
  py::buffer_info bufS0 = sigma_path0.request();
  double *ptrS0 = (double *) bufS0.ptr;
  size_t numS0 = (unsigned int) bufS0.shape[0];
  size_t size = (unsigned int) bufS0.shape[1];
  assert (N == size);
  //sigma_path1
  py::array_t<double> sigma_path1 = py::array_t<double>(bufS0.shape);
	py::buffer_info bufS1 = sigma_path1.request();
  double *ptrS1 = (double *) bufS1.ptr;
  //net
  py::buffer_info bufNet = net.request();
  double *ptrNet = (double *) bufNet.ptr;
  unsigned int Np = (unsigned int) bufNet.shape[1];
  unsigned int Npp = (unsigned int)  bufNet.shape[0];
  //py::print("N",N);
  //py::print("Np",Np);
  //py::print("Npp",Npp);
  assert (N == Np);  // change line 11 # define N 14
  assert (N == Npp);

  //// print net
  //for (unsigned int j=0; j<N;++j){
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrNet[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
	//}

  //py::print("ptrS0\n");
  //for (unsigned int j=0; j<3;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrS0[i+j*N]);
  //  }
  //py::print(j,line);
  //}

  //trans(ptrS0, ptrS1, ptrNet, numS0, N, typ, thr, signFuncInZero);
  unsigned int numX = numS0;
  double Ypred[numX*N];
  unsigned int i;
  //for(i=0; i<numX*N; i++){
  //  Ypred[i] = 0;
  //}
  transCBLAS(ptrS0, Ypred, ptrNet, numS0, N, typ, thr, signFuncInZero);
  for(i=0; i<numX*N; i++){
    ptrS1[i]=Ypred[i];
  }

  //py::print("ptrS1\n");
  //for (unsigned int j=0; j<3;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrS1[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
  //}
  
  return  sigma_path1;
}

void gradientDescentStep(int* ptrY, int* ptrX, unsigned int numX, double* ptrNet0, double* ptrDeltas, double* ptrUpdate, int typ, double thr, int signFuncInZero, int autapse = 0){
  unsigned int i,j,k;

  //py::array_t<int> Ypred = py::array_t<int>(N*numX);
	//py::buffer_info bufYpred = Ypred.request();
  //int *ptrYpred = (int *) bufYpred.ptr;
  
  //py::print("ptrX");
  //for (unsigned int j=0; j<5;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrX[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
  //}
  
  //py::print("ptrNet0");
  //for (unsigned int j=0; j<5;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrNet0[i+j*N]);
  //  }
  //py::print(j,line);
  //}
  //py::print("pars", numX, N,typ, thr, signFuncInZero);
  
  int Ypred[numX*N];
  trans(ptrX, Ypred, ptrNet0, numX, N,typ, thr, signFuncInZero);
  //py::print("Ypred");
  //for (unsigned int j=0; j<5;++j){//numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(Ypred[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
  //}
  for(i=0; i<numX*N; i++){
    ptrDeltas[i] = ptrY[i] - Ypred[i];
  } 

  for(j=0; j<N; j++){
    for(i=0; i<N; i++){
      ptrUpdate[i+N*j] = 0;
      for(k=0;k<numX;k++){
        ptrUpdate[i+N*j] += ptrDeltas[i+N*k] * ptrX[j+N*k];
      }
    }
  }
  
  py::print("autapse",autapse);
  if (autapse == 0){
    for(i=0; i<N; i++)
      ptrUpdate[i+N*i] = 0;
  }
  
  
  
}


py::dict gradientDescentStepCpp(py::array_t<int> Y,py::array_t<int> X, py::array_t<double>net0, int typ = 1,double thr = 0.0, int signFuncInZero = 1, int autapse = 0){
  
  py::dict results;

  //X
  py::buffer_info bufX = X.request();
  int *ptrX = (int *) bufX.ptr;
  size_t numX = (unsigned int) bufX.shape[0];
  size_t size = (unsigned int) bufX.shape[1];
  assert (N == size);
  //Y
  py::buffer_info bufY = Y.request();
  int *ptrY = (int *) bufY.ptr;
  size_t numY = (unsigned int) bufY.shape[0];
  size_t sizeY = (unsigned int) bufY.shape[1];
  assert (N == sizeY);
  assert (numX == numY);
  ////net0
  py::buffer_info bufNet0 = net0.request();
  double *ptrNet0 = (double *) bufNet0.ptr;
  unsigned int N0p = (unsigned int) bufNet0.shape[1];
  unsigned int N0pp = (unsigned int)  bufNet0.shape[0];
  assert (N == N0p);  // change line 11 # define N 14
  assert (N == N0pp);  

  //deltas
  py::array_t<double> deltas = py::array_t<double>(bufX.shape);
  py::buffer_info bufDeltas = deltas.request();
  double *ptrDeltas = (double *) bufDeltas.ptr;   
  //py::print("deltas size",bufDeltas.shape); 
  //update
  py::array_t<double> update = py::array_t<double>(bufNet0.shape);
  py::buffer_info bufUpdate = update.request();
  double *ptrUpdate = (double *) bufUpdate.ptr;
  
  //do stuff
  gradientDescentStep(ptrY,ptrX,numX,ptrNet0,ptrDeltas,ptrUpdate,typ, thr, signFuncInZero,autapse);
  
  results["update"] = update;
  results["deltas"] = deltas;
  return results;
}


void gradientDescentStepCBlasCode(double* ptrY, double* ptrX, unsigned int numX, double* ptrNet0, double* ptrDeltas, double* ptrUpdate, int typ, double thr, int signFuncInZero, int autapse = 0){
  unsigned int i;

  //py::array_t<int> Ypred = py::array_t<int>(N*numX);
	//py::buffer_info bufYpred = Ypred.request();
  //int *ptrYpred = (int *) bufYpred.ptr;

  //py::print("ptrX CBlas");
  //for (unsigned int j=0; j<5;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrX[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
  //}

  //py::print("ptrNet0 CBlas");
  //for (unsigned int j=0; j<5;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrNet0[i+j*N]);
  //  }
  //py::print(j,line);
  //}
  //py::print("pars", numX, N,typ, thr, signFuncInZero);

  double Ypred[numX*N];
  //for(i=0; i<numX*N; i++){
  //  Ypred[i] = 0;
  //}
  //trans(ptrX, Ypred, ptrNet0, numX, N,typ, thr, signFuncInZero);
  transCBLAS(ptrX,Ypred, ptrNet0, numX , N, thr, thr, signFuncInZero);

  //py::print("Ypred CBlas");
  //for (unsigned int j=0; j<5;++j){ //numX
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(Ypred[i+j*N]);
  //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //  }
  //py::print(j,line);
  //}
  for(i=0; i<numX*N; i++){
    ptrDeltas[i] = ptrY[i] - Ypred[i];
  }

  //for(j=0; j<N; j++){
  //  for(i=0; i<N; i++){
  //    ptrUpdate[i+N*j] = 0;
  //    for(k=0;k<numX;k++){
  //      ptrUpdate[i+N*j] += ptrDeltas[i+N*k] * ptrX[j+N*k];
  //    }
  //  }
  //}
  
  //py::print("ptrUpdate cpp");
  //for (unsigned int j=0; j<5;++j){ //N
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(ptrUpdate[i+j*N]);
  //  }
  //py::print(j,line);
  //}
  
  int l=N; //number of rows in C, number of rows in A
  int n=N; //number of columns in C, number of row in B
  int m=numX; //number of columns in A, number of columns in B
  //double update[N*N];
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, l, m, 1.0, ptrX, l, ptrDeltas, l,  0.0, ptrUpdate, l);
  
  //py::print("update CBlas");
  //for (unsigned int j=0; j<5;++j){ //N
  //  py::list line;
  //  for (unsigned int i = 0; i < N; ++i) {
  //    line.append(update[i+j*N]);
  //  }
  //py::print(j,line);
  //} 
  
  py::print("autapse",autapse);
  if (autapse == 0){
    for(i=0; i<N; i++)
      ptrUpdate[i+N*i] = 0;
  }
  
}

py::dict gradientDescentStepCblas(py::array_t<double> Y,py::array_t<double> X, py::array_t<double>net0, int typ = 1,double thr = 0.0, int signFuncInZero = 1, int autapse = 0){
  
  py::dict results;

  //X
  py::buffer_info bufX = X.request();
  double *ptrX = (double *) bufX.ptr;
  size_t numX = (unsigned int) bufX.shape[0];
  size_t size = (unsigned int) bufX.shape[1];
  assert (N == size);
  //Y
  py::buffer_info bufY = Y.request();
  double *ptrY = (double *) bufY.ptr;
  size_t numY = (unsigned int) bufY.shape[0];
  size_t sizeY = (unsigned int) bufY.shape[1];
  assert (N == sizeY);
  assert (numX == numY);
  ////net0
  py::buffer_info bufNet0 = net0.request();
  double *ptrNet0 = (double *) bufNet0.ptr;
  unsigned int N0p = (unsigned int) bufNet0.shape[1];
  unsigned int N0pp = (unsigned int)  bufNet0.shape[0];
  assert (N == N0p);  // change line 11 # define N 14
  assert (N == N0pp);  

  //deltas
  py::array_t<double> deltas = py::array_t<double>(bufX.shape);
  py::buffer_info bufDeltas = deltas.request();
  double *ptrDeltas = (double *) bufDeltas.ptr;   
  //py::print("deltas size",bufDeltas.shape); 
  //update
  py::array_t<double> update = py::array_t<double>(bufNet0.shape);
  py::buffer_info bufUpdate = update.request();
  double *ptrUpdate = (double *) bufUpdate.ptr;
  
  //do stuff
  gradientDescentStepCBlasCode(ptrY,ptrX,numX,ptrNet0,ptrDeltas,ptrUpdate,typ, thr, signFuncInZero, autapse);
  
  results["update"] = update;
  results["deltas"] = deltas;
  return results;
}

void gradientDescentNSteps(int* ptrY,int* ptrX, unsigned int numX,
                             double* ptrNet0, double* ptrNet1, double* ptrDeltas,
                             double alpha,unsigned int NSteps, 
                             int typ, double thr, int signFuncInZero){
  unsigned int step,i,j,k;

  //py::array_t<int> Ypred = py::array_t<int>(N*numX);
	//py::buffer_info bufYpred = Ypred.request();
  //int *ptrYpred = (int *) bufYpred.ptr;
  int Ypred[numX*N];
  double update[N*N];
  for(i=0; i<N*N; i++){ 
    ptrNet1[i] = ptrNet0[i];
  }

  py::print("NSteps",NSteps);
  for(step=0; step<NSteps; step++){
    trans(ptrX, Ypred, ptrNet1, numX, N,typ, thr, signFuncInZero);
    for(i=0; i<numX*N; i++){
      ptrDeltas[i] = ptrY[i] - Ypred[i];
    } 

    for(j=0; j<N; j++){
      for(i=0; i<N; i++){
        update[i+N*j] = 0;
        for(k=0;k<numX;k++){
          update[i+N*j] += ptrDeltas[j+N*k] * ptrX[i+N*k];
        }
      }
    }
    //py::print("cpp net [0]",ptrNet1[0],"alpha",alpha,"update",update[0]); 
    //py::print("cpp net [1]",ptrNet1[1],"alpha",alpha,"update",update[1]);
    //py::print("cpp net [N]",ptrNet1[N],"alpha",alpha,"update",update[N]);  
    for(j=0; j<N; j++){
      for(i=0; i<N; i++){ 
        ptrNet1[i+N*j] += alpha * update[i+N*j];
      }
    }
    //py::print("net1",ptrNet1[0]);
  }
}

py::dict gradientDescentNStepsCpp(py::array_t<int> Y,py::array_t<int> X, py::array_t<double>net0, double alpha, unsigned int NSteps, int typ = 1,double thr = 0.0, int signFuncInZero = 1){
  
  py::dict results;

  //X
  py::buffer_info bufX = X.request();
  int *ptrX = (int *) bufX.ptr;
  size_t numX = (unsigned int) bufX.shape[0];
  size_t size = (unsigned int) bufX.shape[1];
  assert (N == size);
  //Y
  py::buffer_info bufY = Y.request();
  int *ptrY = (int *) bufY.ptr;
  size_t numY = (unsigned int) bufY.shape[0];
  size_t sizeY = (unsigned int) bufY.shape[1];
  assert (N == sizeY);
  assert (numX == numY);
  ////net0
  py::buffer_info bufNet0 = net0.request();
  double *ptrNet0 = (double *) bufNet0.ptr;
  unsigned int N0p = (unsigned int) bufNet0.shape[1];
  unsigned int N0pp = (unsigned int)  bufNet0.shape[0];
  assert (N == N0p);  // change line 11 # define N 14
  assert (N == N0pp);  

  //deltas
  py::array_t<double> deltas = py::array_t<double>(bufX.shape);
  py::buffer_info bufDeltas = deltas.request();
  double *ptrDeltas = (double *) bufDeltas.ptr;   
  py::print("deltas size",bufDeltas.shape); 
  //update
  py::array_t<double> net1 = py::array_t<double>(bufNet0.shape);
  py::buffer_info bufNet1 = net1.request();
  double *ptrNet1= (double *) bufNet1.ptr;
  
  //do stuff
  gradientDescentNSteps(ptrY,ptrX,numX,ptrNet0,ptrNet1,ptrDeltas,alpha,NSteps,typ, thr, signFuncInZero);
  
  results["net"] = net1;
  results["deltas"] = deltas;
  return results;
}



void gradientDescentNStepsCblasCode(double* ptrY,double* ptrX, unsigned int numX,
                             double* ptrNet0, double* ptrNet1, double* ptrDeltas,
                             double alpha,unsigned int NSteps, 
                             int typ, double thr, int signFuncInZero){
  unsigned int step,i,j;

  //py::array_t<int> Ypred = py::array_t<int>(N*numX);
	//py::buffer_info bufYpred = Ypred.request();
  //int *ptrYpred = (int *) bufYpred.ptr;
  double update[N*N];
  for(i=0; i<N*N; i++){ 
    ptrNet1[i] = ptrNet0[i];
  }

  py::print("NSteps",NSteps);
  for(step=0; step<NSteps; step++){
    //gradientDescentStepCBlasCode(ptrY,ptrX,numX,ptrNet1,ptrDeltas,update,typ, thr, signFuncInZero);    
    double Ypred[numX*N];
    //for(i=0; i<numX*N; i++){
    //  Ypred[i] = 0;
    //}
    //trans(ptrX, Ypred, ptrNet0, numX, N,typ, thr, signFuncInZero);
    transCBLAS(ptrX,Ypred, ptrNet1, numX , N, thr, thr, signFuncInZero);

    //py::print("Ypred CBlas");
    //for (unsigned int j=0; j<5;++j){ //numX
    //  py::list line;
    //  for (unsigned int i = 0; i < N; ++i) {
    //    line.append(Ypred[i+j*N]);
    //    //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
    //  }
    //py::print(j,line);
    //}
    for(i=0; i<numX*N; i++){
      ptrDeltas[i] = ptrY[i] - Ypred[i];
    }
    
    //for(j=0; j<N; j++){
    //  for(i=0; i<N; i++){
    //    ptrUpdate[i+N*j] = 0;
    //    for(k=0;k<numX;k++){
    //      ptrUpdate[i+N*j] += ptrDeltas[i+N*k] * ptrX[j+N*k];
    //    }
    //  }
    //}
    
    //py::print("ptrUpdate cpp");
    //for (unsigned int j=0; j<5;++j){ //N
    //  py::list line;
    //  for (unsigned int i = 0; i < N; ++i) {
    //    line.append(ptrUpdate[i+j*N]);
    //  }
    //py::print(j,line);
    //}
    
    int l=N; //number of rows in C, number of rows in A
    int n=N; //number of columns in C, number of row in B
    int m=numX; //number of columns in A, number of columns in B
    //double update[N*N];
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, l, m, 1.0, ptrDeltas, l, ptrX, l,  0.0, update, l);
  
    //py::print("update CBlas");
    //for (unsigned int j=0; j<5;++j){ //N
    //  py::list line;
    //  for (unsigned int i = 0; i < N; ++i) {
    //    line.append(update[i+j*N]);
    //  }
    //py::print(j,line);
    //} 
    
    
    //py::print("cblas net [0]",ptrNet1[0],"alpha",alpha,"update",update[0]); 
    //py::print("cblas net [1]",ptrNet1[1],"alpha",alpha,"update",update[1]);
    //py::print("cblas net [N]",ptrNet1[N],"alpha",alpha,"update",update[N]);  
    for(j=0; j<N; j++){
      for(i=0; i<N; i++){ 
        ptrNet1[i+N*j] += alpha * update[i+N*j];
      }
    }
    //py::print("net1",ptrNet1[0]);
  }
}

py::dict gradientDescentNStepsCblas(py::array_t<double> Y,py::array_t<double> X, py::array_t<double>net0, double alpha, unsigned int NSteps, int typ = 1,double thr = 0.0, int signFuncInZero = 1){
  
  py::dict results;

  //X
  py::buffer_info bufX = X.request();
  double *ptrX = (double *) bufX.ptr;
  size_t numX = (unsigned int) bufX.shape[0];
  size_t size = (unsigned int) bufX.shape[1];
  assert (N == size);
  //Y
  py::buffer_info bufY = Y.request();
  double *ptrY = (double *) bufY.ptr;
  size_t numY = (unsigned int) bufY.shape[0];
  size_t sizeY = (unsigned int) bufY.shape[1];
  assert (N == sizeY);
  assert (numX == numY);
  ////net0
  py::buffer_info bufNet0 = net0.request();
  double *ptrNet0 = (double *) bufNet0.ptr;
  unsigned int N0p = (unsigned int) bufNet0.shape[1];
  unsigned int N0pp = (unsigned int)  bufNet0.shape[0];
  assert (N == N0p);  // change line 11 # define N 14
  assert (N == N0pp);  

  //deltas
  py::array_t<double> deltas = py::array_t<double>(bufX.shape);
  py::buffer_info bufDeltas = deltas.request();
  double *ptrDeltas = (double *) bufDeltas.ptr;   
  py::print("deltas size",bufDeltas.shape); 
  //update
  py::array_t<double> net1 = py::array_t<double>(bufNet0.shape);
  py::buffer_info bufNet1 = net1.request();
  double *ptrNet1= (double *) bufNet1.ptr;
  
  //do stuff
  gradientDescentNStepsCblasCode(ptrY,ptrX,numX,ptrNet0,ptrNet1,ptrDeltas,alpha,NSteps,typ, thr, signFuncInZero);
  
  results["net"] = net1;
  results["deltas"] = deltas;
  return results;
}

PYBIND11_MODULE(hrnn, m) {
    m.doc() = "run recurrent hopfield neural network"; // optional module docstring
    m.def("runRHNN", &runRHNN, "Runs recurrent neural network");
    m.def("runRHNNwithNets", &runRHNNwithNets, "Runs recurrent neural network with a array of connettivity matrices");
    m.def("stateIndex2stateVec", &stateIndex2stateVec, "Returns the binary vector sigma_0 that corresponds to the index m");
    m.def("stateVec2stateIndex", &stateVec2stateIndex, "Returns the index m that corresponds to the binary vector sigma_0");
    m.def("tranCpp", &tranCpp, "transiton function");
    m.def("transManyStatesCpp", &transManyStatesCpp, "transiton function for array of states");
    m.def("transManyStatesCBlas", &transManyStatesCBlas, "transiton function for array of states");
    m.def("gradientDescentStepCpp", &gradientDescentStepCpp, "gradien descent step");
    m.def("gradientDescentStepCblas", &gradientDescentStepCblas, "gradien descent step");
    m.def("gradientDescentNStepsCpp", &gradientDescentNStepsCpp, "gradien descent N step");
    m.def("gradientDescentNStepsCblas", &gradientDescentNStepsCblas, "gradien descent N step");
}
