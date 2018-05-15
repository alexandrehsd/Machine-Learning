// referência http://eric-yuan.me/cpp-read-mnist/
// remark 01: na criação de ifstream file(...) não passe o nome do arquivo por uma variável string
// remark 02: coloquei o arquivo do mnist na mesma pasta do código, por isso não escrevo o caminho, só deu certo assim
// Link para a normalização dos pesos http://www.iro.umontreal.ca/~bengioy/ift6266/H12/html/mlp_en.html

#include<math.h>
#include<iostream>
#include<vector>
#include<fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>

#define NL1 450 //Number of neurons of the first hidden layer
#define NL2 10 //Number of neurons of the second hidden layer (also, length of the output vector)
#define NoI 40000 //Number of images used to train the algorithm
#define eta 0.2 //passo de aprendizagem
using namespace std;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return (((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4);
}

void read_Mnist_train(vector< vector<float> > &images)
{
    ifstream file ("train-images.idx3-ubyte", ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        //Para ler todas as imagens, executar:
        // for(int i = 0; i < number_of_images; ++i)

        for(int i = 0; i < NoI; ++i)
        {
            vector<float> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((float)temp)/2550);//entrada normalizada
                }
            }
            images.push_back(tp);
        }
    }
}

void read_Mnist_sample(vector< vector<float> > &images)
{
    ifstream file ("t10k-images.idx3-ubyte", ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        //Para ler todas as imagens, executar:
        // for(int i = 0; i < number_of_images; ++i)

        for(int i = 0; i < 5000; ++i)
        {
            vector<float> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((float)temp)/2550);//entrada normalizada
                }
            }
            images.push_back(tp);
        }
    }
}

void read_Mnist_Label_train(vector<int> &labels)
{
    ifstream file ("train-labels.idx1-ubyte", ios::binary);

    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        // Para ler todas as imagens, executar:
        // for(int i = 0; i < number_of_images; ++i)
        for(int i = 0; i < NoI; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels[i] = (int)temp;
        }
    }
}

void read_Mnist_Label_sample(vector<int> &labels)
{
    ifstream file ("t10k-labels.idx1-ubyte", ios::binary);

    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        // Para ler todas as imagens, executar:
        // for(int i = 0; i < number_of_images; ++i)
        for(int i = 0; i < 5000; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels[i] = (int)temp;
        }
    }
}

//Função sigmoide
double fz(double z){
    return 1/(1 + exp(-z));
}

//Derivada da função sigmoide
double dfz(double z){
    return fz(z)*(1 - fz(z));
}

//Função para gerar os pesos da primeira camada aleatoriamente 
void weightsLayer1Gen(double (&wlayer1)[NL1][785]){
    for(int i=0;i<NL1;i++){
        for(int j=0;j<785;j++){
            wlayer1[i][j] = (((double) rand() / (RAND_MAX))/7.364) - 0.0679;//peso normalizado [-0.0679,0.0679]
        }
    }
}

//Função para calcular os Z's da primeira camada
void Zlayer1(double (&zvec1)[NL1], double (&wlayer1)[NL1][785], vector<float> inputs){
    float sum = 0;
    for (int i = 0; i < NL1; i++)
    {
        sum += wlayer1[i][0]*(-1); //bias
        for (int j = 1; j < 785; j++)
        {
            sum += wlayer1[i][j]*inputs[j-1];
        }
        zvec1[i] = sum;
        sum = 0;
    }
}

//Função para calcular as saídas da primeira camada (entradas da segunda)
void inpLayer2(double (&inp2)[NL1], double (&zvec1)[NL1]){
    for (int i = 0; i < NL1; ++i)
    {
        inp2[i] = fz(zvec1[i]);
    }
}

//Função para gerar os pesos da segunda camada aleatoriamente 
void weightsLayer2Gen(double (&wlayer2)[NL2][NL1 + 1]){
    for(int i=0;i<NL2;i++){
        for(int j=0;j<NL1+1;j++){
            wlayer2[i][j] = (((double) rand() / (RAND_MAX))/4.378) - 0.1142;//peso normalizado [-0.1142,0.1142]
        }
    }
}

//Função para calcular os Z's da segunda camada
void Zlayer2(double (&zvec2)[NL2], double (&wlayer2)[NL2][NL1+1], double (&inp2)[NL1]){
    for (int i = 0; i < NL2; i++)
    {
        float sum = 0;
        sum += wlayer2[i][0]*(-1); //bias
        for (int j = 1; j < NL1+1; j++)
        {
            sum += wlayer2[i][j]*inp2[j-1];
        }
        zvec2[i] = sum;
    }
}

//Função para calcular as saídas da rede
void output(double (&out)[NL2], double (&zvec2)[NL2]){
    for (int i = 0; i < NL2; ++i)
    {
        out[i] = fz(zvec2[i]);
    }
}

//Função para atualizar a matriz de erros
void errUpdate(double (&errvec)[NL2], double (&out)[NL2], vector<int> &labels, int count){
    float expvec[NL2] = {0,0,0,0,0,0,0,0,0,0};
    expvec[labels[count]] = 1; //Definindo qual a saída esperada
    for (int i = 0; i < NL2; ++i)
    {
        errvec[i] = expvec[i]-out[i]; //definindo qual o vetor de erros para a imagem "count"
    }
}

//Função para atualizar os valores do gradiente da camada 2
void deltaL2(double (&gradL2)[NL2], double (&out)[NL2], double (&errvec)[NL2]){
    for (int i = 0; i < NL2; ++i)
    {
        gradL2[i] = out[i]*(1 - out[i])*errvec[i]; 
    }
}

//Função para atualizar os valores do gradiente da camada 2
void deltaL1(double (&gradL1)[NL1], double (&inp2)[NL1], double (&wlayer2)[NL2][NL1+1], 
    double (&gradL2)[NL2]){
    
    for (int i = 0; i < NL1; ++i)
    {
        float sum = 0;
        for (int j = 0; j < NL2; ++j)
        {
            sum += gradL2[j]*wlayer2[j][i+1];
        }
        gradL1[i] = inp2[i]*(1 - inp2[i])*sum;
    }
}

// wlayer2 [10][393]
//Função para atualizar os pesos da camada 2
void weightsLayer2Updt(double (&wlayer2)[NL2][NL1+1], double (&gradL2)[NL2], double (&inp2)[NL1]){
    for (int i = 0; i < NL2; ++i)
    {
        for (int j = 0; j < NL1+1; ++j)
        {
            wlayer2[i][j] = wlayer2[i][j] + eta*(gradL2[i] * inp2[j]);
        }
    }
}

//Função para atualizar os pesos da camada 1
void weightsLayer1Updt(double (&wlayer1)[NL1][785], double (&gradL1)[NL1], vector<float> &images){
    for (int i = 0; i < NL1; ++i)
    {
        for (int j = 0; j < 785; ++j)
        {
            wlayer1[i][j] = wlayer1[i][j] + eta*(gradL1[i]*images[j]); 
        }
    }
}

int NetOut(vector<float> sample, double (&wlayer1)[NL1][785], double (&wlayer2)[NL2][NL1+1]){
    double zvector1[NL1];
    
    for (int i = 0; i < NL1; i++)
    {
        double sum = 0;
        sum += wlayer1[i][0]*(-1); //bias
        for (int j = 1; j < 785; j++)
        {
            sum += wlayer1[i][j]*sample[j-1];
        }
        zvector1[i] = sum;
    }

    double inputLayer2[NL1];
    for (int i = 0; i < NL1; ++i)
    {
        inputLayer2[i] = fz(zvector1[i]); 
    }

    double zvector2[NL2];
    for (int i = 0; i < NL2; i++)
    {
        double sum = 0;
        sum += wlayer2[i][0]*(-1); //bias
        for (int j = 1; j < NL1+1; j++)
        {
            sum += wlayer2[i][j]*inputLayer2[j-1];
        }
        zvector2[i] = sum;
    }

    double Netoutput[NL2];
    double greater = 0;
    int index;
    for (int i = 0; i < NL2; ++i)
    {
        Netoutput[i] = fz(zvector2[i]);

        if (Netoutput[i] > greater)
        {
            greater = Netoutput[i];
            index = i;
        }
    }
    return index;
}

int main(int argc, char const *argv[])
{

    int image_size = 28 * 28;
    srand( (unsigned)time( NULL ) );

    //read MNIST image into float vector
    vector<vector<float> > images; //input of the Layer 1
    read_Mnist_train(images);
    cout<<images.size()<<endl;
    cout<<images[0].size()<<endl;

    //read MNIST label into float vector
    vector<int> labels(NoI);
    read_Mnist_Label_train(labels);
    cout<<labels.size()<<endl;

    vector<vector<float> > samples;
    read_Mnist_sample(samples);

    vector<int> lsamples(5000);
    read_Mnist_Label_sample(lsamples);

    //Declarando e inicializando a matriz de pesos da camada 1
    double wlayer1[NL1][785];
    weightsLayer1Gen(wlayer1);

    //Declarando e inicializando os pesos da camada 2
    double wlayer2[NL2][NL1+1];
    weightsLayer2Gen(wlayer2);

    //Declarando os Z's da camada 1 e 2
    double zvec1[NL1];
    double zvec2[NL2];

    //Declarando o vetor de entradas e saídas da camada 2
    double inp2[NL1];
    double out[NL2];

    //Declarando as matrizes dos gradientes
    double gradL2[NL2];
    double gradL1[NL1];

    //Declarando a matriz de erros
    double errvec[NL2];

    for (int epoch = 0; epoch < 7; ++epoch)
    {
        for (int count = 0; count < NoI; ++count)
        {
            //inicializando os Z's da camada 1
            Zlayer1(zvec1,wlayer1,images[count]);

            //gerando os inputs da camada 2
            inpLayer2(inp2,zvec1);

            //inicializando o vetor de Z's da camada 2
            Zlayer2(zvec2,wlayer2,inp2);

            //Recebendo as saídas da rede
            output(out,zvec2);

            //Atualizando o vetor de erro da imagem[count]
            errUpdate(errvec, out, labels, count);

            //Atualizando o vetor de gradientes da layer 2
            deltaL2(gradL2, out, errvec);

            //Atualizando os pesos da camada 2
            weightsLayer2Updt(wlayer2,gradL2,inp2);

            //Atualizando o vetor de gradientes da layer 2
            deltaL1(gradL1, inp2, wlayer2, gradL2);

            //Atualizando os pesos da camada 1
            weightsLayer1Updt(wlayer1,gradL1,images[count]);
            if (count % 10000 == 0)
            {

                cout << "vetor de saidas: ";
                for (int i = 0; i < NL2; ++i)
                {
                    cout << out[i] << " ";
                }
                cout << endl;

                float expvec[NL2] = {0,0,0,0,0,0,0,0,0,0};
                expvec[(int)labels[count]] = 1; //Definindo qual a saída esperada

                cout << "Saída esperada: ";
                for (int i = 0; i < NL2; ++i)
                {
                    cout << expvec[i] << " ";
                }
                cout << endl;

                float sum = 0;
                cout << "vetor de erros: ";
                for (int i = 0; i < NL2; ++i)
                {
                    cout << errvec[i] << " ";
                }
                cout << endl;

                cout << "Z's da camada 2: ";
                for (int i = 0; i < NL2; ++i)
                {
                    cout << zvec2[i] << " ";
                }
                cout << endl;

                cout << "Z's da camada 1: ";
                for (int i = 0; i < NL2; ++i)
                {
                    cout << zvec1[i] << " ";
                }
                cout << endl;

                cout << "Label = " << labels[count] << endl;
            }
        }
        cout << endl;
        cout << "A ÉPOCA " << (epoch+1) << " DO TREINAMENTO ESTÁ COMPLETA!" << endl << endl;
    }
    

    int hits = 0;
    int vetHits[10] = {};
    int qtds[10] = {};
    float taxaAcerto[10];
    for (int i = 0; i < 5000; ++i)
    {

        // cout << "Label de entrada = " << lsamples[i] << endl;
        int saida = NetOut(samples[i], wlayer1, wlayer2);
        // cout << "Label de Saída   = " << saida << endl;
        qtds[lsamples[i]]++;
        if (lsamples[i] == saida)
        {
            hits++;
            vetHits[saida]++;
        }
    }
    cout << "Taxa de acerto = " << (hits/5000.0)*100 << "%" << endl;
    cout << "Taxa de acerto de cada digito: " << endl;
    for(int i=0; i<10; i++){
        cout << "Digito " << i << ": " << (float(vetHits[i])/qtds[i])*100 << "%" << endl;
    }
    
    return 0;
}
