// referência https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
// remark 01: na criação de ifstream file(...) não passe o nome do arquivo por uma variável string
// remark 02: eu coloquei o arquivo do mnist na mesma pasta do código, por isso não mostro o caminho

#include <iostream>
#include <vector>
#include <fstream>
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

void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector< vector<double> > &arr)
{   
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage)); 
    
    ifstream file;
    file.open ("t10k-images.idx3-ubyte",ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        cout << "AQUI";
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        cout << "magic_number: " << magic_number << endl;

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        cout << "number_of_images: " << number_of_images << endl;

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        cout << "n_rows: " << n_rows << endl;

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        cout << "ncows: " << n_cols << endl;

        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
    file.close();
}
 
int main()
{
    vector< vector<double> > ar;
    ReadMNIST(10000,784,ar);

    cout<<ar.size()<<endl;
    cout<<ar[0].size()<<endl;

    for (int i = 15; i < 16; i++)
    {
        cout << "----------------------- "<< "image" << i << " ------------------------------" << endl;
        for (int j = 0; j < ar[i].size(); j++)
        {
            cout << "Position " << j << ": " << ar[i][j] <<endl;
        }
    }

  return 0;
}