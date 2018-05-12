// referência http://eric-yuan.me/cpp-read-mnist/
// remark 01: na criação de ifstream file(...) não passe o nome do arquivo por uma variável string
// remark 02: coloquei o arquivo do mnist na mesma pasta do código, por isso não escrevo o caminho, só deu certo assim

#include<math.h>
#include<iostream>
#include<vector>
#include<fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>

#define NL1 392 //Number of neurons from the first hidden layer
#define NL2 10 //Number of neurons from the second hidden layer (also, length of the output vector)
#define NoI 20000 //Number of images used to train the algorithm
#define eta 0.6 //passo de aprendizagem
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

        for(int i = 0; i < 10; ++i)
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

void read_Mnist_Label_train(vector<float> &labels)
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
            labels[i] = (float)temp;
        }
    }
}

void read_Mnist_Label_sample(vector<float> &labels)
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
        for(int i = 0; i < 10; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels[i] = (float)temp;
        }
    }
}

//Função sigmoide
float fz(float z){
    return 1/(1 + exp(-z));
}

//Derivada da função sigmoide
float dfz(float z){
    return fz(z)*(1 - fz(z));
}

//Função para gerar os pesos da primeira camada aleatoriamente 
void weightsLayer1Gen(float (&wlayer1)[NL1][785]){
    for(int i=0;i<NL1;i++){
        for(int j=0;j<785;j++){
            wlayer1[i][j] = ((float) rand() / (RAND_MAX))/10;//peso normalizado
        }
    }
}

//Função para calcular os Z's da primeira camada
void Zlayer1(float (&zvec1)[NL1], float (&wlayer1)[NL1][785], vector<float> inputs){
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
void inpLayer2(float (&inp2)[NL1], float (&zvec1)[NL1]){
    for (int i = 0; i < NL1; ++i)
    {
        inp2[i] = fz(zvec1[i]);
    }
}

//Função para gerar os pesos da segunda camada aleatoriamente 
void weightsLayer2Gen(float (&wlayer2)[NL2][NL1 + 1]){
    for(int i=0;i<NL2;i++){
        for(int j=0;j<NL1+1;j++){
            wlayer2[i][j] = ((float) rand() / (RAND_MAX))/10;//peso normalizado
        }
    }
}

//Função para calcular os Z's da segunda camada
void Zlayer2(float (&zvec2)[NL2], float (&wlayer2)[NL2][NL1+1], float (&inp2)[NL1]){
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
void output(float (&out)[NL2], float (&zvec2)[NL2]){
    for (int i = 0; i < NL2; ++i)
    {
        out[i] = fz(zvec2[i]);
    }
}

//Função para atualizar a matriz de erros
void errUpdate(float (&errvec)[NL2], float (&out)[NL2], vector<float> &labels, int count){
    float expvec[NL2] = {0,0,0,0,0,0,0,0,0,0};
    expvec[(int)labels[count]] = 1; //Definindo qual a saída esperada
    for (int i = 0; i < NL2; ++i)
    {
        errvec[i] = expvec[i]-out[i]; //definindo qual o vetor de erros para a imagem "count"
    }
}

//Função para atualizar os valores do gradiente da camada 2
void deltaL2(float (&gradL2)[NL2], float (&out)[NL2], float (&errvec)[NL2]){
    for (int i = 0; i < NL2; ++i)
    {
        gradL2[i] = out[i]*(1 - out[i])*errvec[i]; 
    }
}

//Função para atualizar os valores do gradiente da camada 2
void deltaL1(float (&gradL1)[NL1], float (&inp2)[NL1], float (&wlayer2)[NL2][NL1+1], 
    float (&gradL2)[NL2]){
    
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
void weightsLayer2Updt(float (&wlayer2)[NL2][NL1+1], float (&gradL2)[NL2], float (&inp2)[NL1]){
    for (int i = 0; i < NL2; ++i)
    {
        for (int j = 0; j < NL1+1; ++j)
        {
            wlayer2[i][j] = wlayer2[i][j] + eta*(gradL2[i] * inp2[j]);
        }
    }
}

//Função para atualizar os pesos da camada 1
void weightsLayer1Updt(float (&wlayer1)[NL1][785], float (&gradL1)[NL1], vector<float> &images){
    for (int i = 0; i < NL1; ++i)
    {
        for (int j = 0; j < 785; ++j)
        {
            wlayer1[i][j] = wlayer1[i][j] + eta*(gradL1[i]*images[j]); 
        }
    }
}

void NetOut(vector<float> sample, float (&wlayer1)[NL1][785], float (&wlayer2)[NL2][NL1+1]){
    float zvector1[NL1];
    
    for (int i = 0; i < NL1; i++)
    {
        float sum = 0;
        sum += wlayer1[i][0]*(-1); //bias
        for (int j = 1; j < 785; j++)
        {
            sum += wlayer1[i][j]*sample[j-1];
        }
        zvector1[i] = sum;
    }

    float inputLayer2[NL1];
    for (int i = 0; i < NL1; ++i)
    {
        inputLayer2[i] = fz(zvector1[i]); 
    }

    float zvector2[NL2];
    for (int i = 0; i < NL2; i++)
    {
        float sum = 0;
        sum += wlayer2[i][0]*(-1); //bias
        for (int j = 1; j < NL1+1; j++)
        {
            sum += wlayer2[i][j]*inputLayer2[j-1];
        }
        zvector2[i] = sum;
    }

    float Netoutput[NL2];
    float greater = 0;
    float index;
    for (int i = 0; i < NL2; ++i)
    {
        Netoutput[i] = fz(zvector2[i]);

        if (Netoutput[i] > greater)
        {
            greater = Netoutput[i];
            index = i;
        }
    }

    cout << "Vetor de saída = ";
    for (int i = 0; i < 10; ++i)
    {
        cout << Netoutput[i] << " ";
    }
    cout << endl;
    cout << "Saída da rede = " << index << endl;
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
    vector<float> labels(NoI);
    read_Mnist_Label_train(labels);
    cout<<labels.size()<<endl;

    vector<vector<float> > samples;
    read_Mnist_sample(samples);

    vector<float> lsamples(10);
    read_Mnist_Label_sample(lsamples);

    //Declarando e inicializando a matriz de pesos da camada 1
    float wlayer1[NL1][785];
    weightsLayer1Gen(wlayer1);

    //Declarando e inicializando os pesos da camada 2
    float wlayer2[NL2][NL1+1];
    weightsLayer2Gen(wlayer2);

    //Declarando os Z's da camada 1 e 2
    float zvec1[NL1];
    float zvec2[NL2];

    //Declarando o vetor de entradas e saídas da camada 2
    float inp2[NL1];
    float out[NL2];

    //Declarando as matrizes dos gradientes
    float gradL2[NL2];
    float gradL1[NL1];

    //Declarando a matriz de erros
    float errvec[NL2];

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
        if (count % 5000 == 0)
        {
            float sum = 0;
            cout << "vetor de erros: ";
            for (int i = 0; i < NL2; ++i)
            {
                cout << errvec[i] << " ";
            }
            cout << endl;

            cout << "vetor de Z's da camada 2: ";
            for (int i = 0; i < NL2; ++i)
            {
                cout << zvec2[i] << " ";
            }
            cout << endl;
        }
    }

    cout <<"Último label = " << labels[NoI-1] << endl;
    cout <<"Última saída = ";
    for (int i = 0; i < 10; ++i)
    {
         cout << out[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < 10; ++i)
    {
        cout << "Label de entrada = " << lsamples[i] << endl;
        NetOut(samples[i], wlayer1, wlayer2);
    }

    return 0;
}
