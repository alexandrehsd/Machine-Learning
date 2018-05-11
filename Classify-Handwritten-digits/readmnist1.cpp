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
#define NoI 10000 //Number of images used to train the algorithm

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

void read_Mnist(vector< vector<float> > &images)
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
                    tp.push_back(((float)temp)/255);//entrada normalizada
	            }
            }
            images.push_back(tp);
        }
    }
}

void read_Mnist_Label(vector<float> &labels)
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
    /* initialize random seed: */
    srand( (unsigned)time( NULL ) );

    for(int i=0;i<NL1;i++){
        for(int j=0;j<785;j++){
            wlayer1[i][j] = ((float) rand() / (RAND_MAX));//peso normalizado
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
    /* initialize random seed: */
    srand( (unsigned)time( NULL ) );

    for(int i=0;i<NL2;i++){
        for(int j=0;j<NL1+1;j++){
            wlayer2[i][j] = ((float) rand() / (RAND_MAX));//peso normalizado
        }
    }
}

//Função para calcular os Z's da segunda camada
void Zlayer2(float (&zvec2)[NL2], float (&wlayer2)[NL2][NL1+1], float (&inp2)[NL1]){
    
    float sum = 0;

    for (int i = 0; i < NL2; i++)
    {
        sum += wlayer2[i][0]*(-1); //bias
        for (int j = 1; j < NL1+1; j++)
        {
            sum += wlayer2[i][j]*inp2[j-1];
        }
        /*O valor de sum estava saindo entre [10,13],
        Então eu estou simplesmente dividindo por 10 aqui
        para não perdermos sensibilidade na função sigmoide,
        este comentário serve para lembrar de consultar esta ação depois.
        zvec2[i] = sum/10;*/
        zvec2[i] = sum;
        sum = 0;
    }
}

//Função para calcular as saídas da rede
void output(float (&out)[NL2], float (&zvec2)[NL2]){
    for (int i = 0; i < NL2; ++i)
    {
        out[i] = fz(zvec2[i]);
    }
}

void expectedOutput(float (&expvec)[NL2],int label){
    switch(label){
        case 0:
            expvec[0] = 1;
            break;
        case 1:
            expvec[1] = 1;
            break;
        case 2:
            expvec[2] = 1;
            break;
        case 3:
            expvec[3] = 1;
            break;
        case 4:
            expvec[4] = 1;
            break;
        case 5:
            expvec[5] = 1;
            break;
        case 6:
            expvec[6] = 1;
            break;
        case 7:
            expvec[7] = 1;
            break;
        case 8:
            expvec[8] = 1;
            break;
        case 9:
            expvec[9] = 1;
            break;
        default: cout << "Label inválido";
    }
}

//Função para atualizar a matriz de erros
void errUpdate(float (&errvec)[NoI][NL2], float (&out)[NL2], float (&labels)[NoI], int count){
    float expvec[NL2] = {0,0,0,0,0,0,0,0,0,0};
    expectedOutput(expvec,(int)labels[count]); //Definindo qual a saída esperada
    for (int i = 0; i < NL2; ++i)
    {
        errvec[count][i] = expvec[i] - out[i]; //definindo qual o vetor de erros para a imagem "count"
    }
}

//Função para atualizar os valores do gradiente da camada 2
void deltaL2(float (&gradL2)[NoI][NL2], float (&out)[NL2], float (&errvec)[NoI][NL2], int count){
    for (int i = 0; i < NL2; ++i)
    {
        gradL2[count][i] = out[i]*(1 - out[i])*errvec[count][i]; 
    }
}

//Função para atualizar os valores do gradiente da camada 2
void deltaL1(float (&gradL1)[NoI][NL1], float (&inp2)[NL1], float (&wlayer2)[NL2][NL1+1], 
    float (&gradL2)[NoI][NL2], int count){
    float sum = 0;
    for (int i = 0; i < NL1; ++i)
    {
        for (int j = 0; j < NL2; ++j)
        {
            sum += gradL2[count][j]*wlayer2[j][i];
        }
        gradL1[count][i] = inp2[i]*(1 - inp2[i])*sum;
        sum = 0;
    }
}

//Função para atualizar os pesos da camada 2
void weightsLayer2Updt(float (&wlayer2)[NL2][NL1+1], float (&gradL2)[NoI][NL2], float (&inp2)[NL1]){
/*    float sum = 0;
    for (int i = 0; i < NL2; ++i)
    {
        
    }*/
}

int main(int argc, char const *argv[])
{

	int number_of_images = NoI;
	int image_size = 28 * 28;

    //read MNIST image into float vector
    vector<vector<float> > images;

    read_Mnist(images);

    cout<<images.size()<<endl;
    cout<<images[0].size()<<endl;

    //read MNIST label into float vector
    vector<float> labels(number_of_images);

    read_Mnist_Label(labels);

    cout<<labels.size()<<endl;

    int count = 0; //contador para armazenar qual o índice da imagem que está entrando na rede

    //Declarando e inicializando a matriz de pesos da camada 1
    float wlayer1[NL1][785];
    weightsLayer1Gen(wlayer1);

    //Declarando e inicializando os Z's da camada 1
    float zvec1[NL1];
    Zlayer1(zvec1,wlayer1,images[0]);

    //Declarando e gerando os inputs da camada 2
    float inp2[NL1];
    inpLayer2(inp2,zvec1);

    //Declarando e inicializando os pesos da camada 2
    float wlayer2[NL2][NL1+1];
    weightsLayer2Gen(wlayer2);

    //Declarando e inicializando o vetor de Z's da camada 2
    float zvec2[NL2];
    Zlayer2(zvec2,wlayer2,inp2);

    //Declarando e recebendo as saídas da rede
    float out[NL2];
    output(out,zvec2);

    //Declarando a matriz de erros (cada imagem tem um vetor de erro associado à sua saída)
    float errvec[NoI][NL2];

    // cout << "Vetor de labels ------------------------" << endl;
    // for (int i = 0; i < NL1; ++i)
    // {
    //     cout << "labels " << i << ": " << labels[i] << endl;
    // }

/*    cout << "Vetor de saida ------------------" << endl;
    for (int i = 0; i < NL2; ++i)
    {
        cout << "out " << i << ": " << out[i] << endl;
    }*/

/*
    cout << "Vetor de inputs -----------------" << endl;

    for (int i = 0; i < 784; ++i)
    {
        cout << images[0][i] << endl;
    }*/

/*    cout << "vetor de pesos ---------------" << endl;

    for (int i = 0; i < 785; i++)
    {
        cout << "Peso " << i << ": "<< wlayer1[391][i] << endl;
    }*/
	return 0;
}
