// referência http://eric-yuan.me/cpp-read-mnist/
// remark 01: na criação de ifstream file(...) não passe o nome do arquivo por uma variável string
// remark 02: coloquei o arquivo do mnist na mesma pasta do código, por isso não escrevo o caminho, só deu certo assim

#include<math.h>
#include<iostream>
#include<vector>
#include<fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#define M 392
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

void read_Mnist(vector< vector<float> > &vec)
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

        for(int i = 0; i < 10000; ++i)
        {
            vector<float> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((float)temp)/2040);//entrada normalizada
	            }
            }
            vec.push_back(tp);
        }
    }
}

void read_Mnist_Label(vector<float> &vec)
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
        for(int i = 0; i < 10000; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i] = (float)temp;
        }
    }
}

float fz(float z){
    return 1/(1 + exp(-z));
}

float dfz(float z){
    return fz(z)*(1 - fz(z));
}

//Função para gerar os pesos da primeira camada aleatoriamente 
void weightsLayer1Gen(float (&wlayer1)[392][785]){
    /* initialize random seed: */
    srand( (unsigned)time( NULL ) );

    for(int i=0;i<392;i++){
        for(int j=0;j<785;j++){
            wlayer1[i][j] = ((float) rand() / (RAND_MAX))/10;//peso normalizado
        }
    }
}

void Zlayer1(float (&zvec1)[392], float wlayer1[392][785], vector<float> inputs){
    
    float sum = 0;
    int v = 392; //Número de neurônios da camada oculta 1
    for (int i = 0; i < v; i++)
    {
        sum += wlayer1[i][0]*(-1); //bias
        /*Há 785 (784 entradas + bias) pesos para cada neuronio,
        porém, há apenas 784 entradas. De modo que,
        desconsiderando o bias, o primeiro peso está na posição
        w[i][1], mas o primeiro input na posição inputs[0].
        Assim, a influencia do bias é calculada antes 
        e a partir daqui é calculado o produto dos pesos
        pelas entradas -> w[0][1 -> peso1]*input[0]...*/
        for (int j = 1; j < 785; j++)
        {
            sum += wlayer1[i][j]*inputs[j-1];
        }
        zvec1[i] = sum;
        sum = 0;
    }
}

void inpLayer2(float (&inp2)[392], float zvec1[392]){
    for (int i = 0; i < 392; ++i)
    {
        inp2[i] = fz(zvec1[i]);
    }
}

//Função para gerar os pesos da segunda camada aleatoriamente 
void weightsLayer2Gen(float (&wlayer2)[50][393]){
    /* initialize random seed: */
    srand( (unsigned)time( NULL ) );

    for(int i=0;i<50;i++){
        for(int j=0;j<393;j++){
            wlayer2[i][j] = ((float) rand() / (RAND_MAX))/10;//peso normalizado
        }
    }
}

void Zlayer2(float (&zvec2)[50], float wlayer2[50][393], float (&inp2)[392]){
    
    float sum = 0;
    int m = 50; //Número de neurônios da camada oculta 2
    for (int i = 0; i < m; i++)
    {
        sum += wlayer2[i][0]*(-1); //bias
        /*Há 393 (392 entradas + bias) pesos para cada neuronio,
        porém, há apenas 392 entradas. Assim, o bias é calculado
        antes e a partir daqui é calculado o produto dos pesos
        pelas entradas -> w[0][1 -> peso1]*input[0]...*/
        for (int j = 1; j < 393; j++)
        {
            sum += wlayer2[i][j]*inp2[j-1];
        }
        /*O valor de sum estava saindo entre [10,13],
        Então eu estou simplesmente dividindo por 10 aqui
        para não perdermos sensibilidade na função sigmoide,
        este comentário serve para lembrar de consultar esta ação depois*/
        zvec2[i] = sum/10;
        sum = 0;
    }
}

void output(float (&out)[50], float (&zvec2)[50]){
    for (int i = 0; i < 50; ++i)
    {
        out[i] = fz(zvec2[i]);
    }
}

int main(int argc, char const *argv[])
{

	int number_of_images = 10000;
	int image_size = 28 * 28;

    //read MNIST image into float vector
    vector<vector<float> > vec1;

    read_Mnist(vec1);

    cout<<vec1.size()<<endl;
    cout<<vec1[0].size()<<endl;

    //read MNIST label into float vector
    vector<float> vec2(number_of_images);

    read_Mnist_Label(vec2);

    cout<<vec2.size()<<endl;

    //Declarando e inicializando a matriz de pesos da camada 1
    float wlayer1[392][785];
    weightsLayer1Gen(wlayer1);

    //Declarando e inicializando os Z's da camada 1
    float zvec1[392];
    Zlayer1(zvec1,wlayer1,vec1[0]);

    //Declarando e gerando os inputs da camada 2
    float inp2[392];
    inpLayer2(inp2,zvec1);

    //Declarando e inicializando os pesos da camada 2
    float wlayer2[50][393];
    weightsLayer2Gen(wlayer2);

    //Declarando e inicializando o vetor de Z's da camada 2
    float zvec2[50];
    Zlayer2(zvec2,wlayer2,inp2);

    //Declarando e recebendo as saídas da rede
    float out[50];
    output(out,zvec2);

/*    cout << "Vetor Z ------------------------" << endl;
    for (int i = 0; i < 392; ++i)
    {
        cout << "Z " << i << ": " << zvec1[i] << endl;
    }*/

/*    cout << "Vetor de inp2 ------------------" << endl;
    for (int i = 0; i < 392; ++i)
    {
        cout << "inp2 " << i << ": " << inp2[i] << endl;
    }*/


/*    cout << "Vetor de inputs -----------------" << endl;

    for (int i = 0; i < 784; ++i)
    {
        cout << vec1[0][i] << endl;
    }*/

/*    cout << "vetor de pesos ---------------" << endl;

    for (int i = 0; i < 785; i++)
    {
        cout << "Peso " << i << ": "<< wlayer1[391][i] << endl;
    }*/


	return 0;
}
