//
// Created by boruoshihao on 5/30/17.
//

#include "../../include/generalHamiltonian_icf/generalHamiltonian_sym_icf.h"

using namespace std;
using namespace H5;
using namespace tensor_hao;

GeneralHamiltonian_sym_icf::GeneralHamiltonian_sym_icf():L(0), N(0) { }

GeneralHamiltonian_sym_icf::GeneralHamiltonian_sym_icf(const string &filename) { read(filename); }

GeneralHamiltonian_sym_icf::~GeneralHamiltonian_sym_icf() { }

size_t GeneralHamiltonian_sym_icf::getL() const { return L; }

// size_t GeneralHamiltonian_sym_icf::getKlx() const { return k_l_x; }

// size_t GeneralHamiltonian_sym_icf::getKly() const { return k_l_y; }

size_t GeneralHamiltonian_sym_icf::getN() const { return N; }

size_t GeneralHamiltonian_sym_icf::getNup() const { return Nup; }

size_t GeneralHamiltonian_sym_icf::getNdn() const { return Ndn; }

size_t GeneralHamiltonian_sym_icf::getSVDNumber() const { return svdNumber; }

const TensorHao<complex<double>,2> &GeneralHamiltonian_sym_icf::getK() const { return K; }

const TensorHao<complex<double>,2> &GeneralHamiltonian_sym_icf::getSVDVecs() const { return svdVecs; }

const TensorHao<complex<double>,1> &GeneralHamiltonian_sym_icf::getSVDBg() const { return svdBg; }

const TensorHao<complex<double>,1> &GeneralHamiltonian_sym_icf::getInitialBg() const { return initialBg; }

size_t GeneralHamiltonian_sym_icf::getKpEigenStatus() const { return KpEigenStatus; }

const TensorHao<complex<double>,2> &GeneralHamiltonian_sym_icf::getKp() const { return Kp; }

const TensorHao<double,1> &GeneralHamiltonian_sym_icf::getKpEigenValue() const { return KpEigenValue; }

const TensorHao<complex<double>, 2> &GeneralHamiltonian_sym_icf::getKpEigenVector() const { return KpEigenVector; }

void GeneralHamiltonian_sym_icf::read(const string &filename)
{
    H5File file(filename, H5F_ACC_RDONLY);
    // readFile(k_l_x, file, "k_l_x");
    // readFile(k_l_y, file, "k_l_y");
    readFile(L, file, "L");
    readFile(Nup, file, "Nup");
    readFile(Ndn, file, "Ndn");
    readFile(N, file, "N");
    readFile(svdNumber, file, "svdNumber");

    TensorHao<double,2> K_r,K_i;
    TensorHao<double,2> svdVecs_r,svdVecs_i;
    TensorHao<double,1> svdBg_r,svdBg_i;
  
    K_r.resize(L, L); readFile( K_r.size(),  K_r.data(),  file, "K_r" );
    K_i.resize(L, L); readFile( K_i.size(),  K_i.data(),  file, "K_i" );
    //Vq is not depends on k
    svdVecs_r.resize(L, svdNumber); readFile(svdVecs_r.size(), svdVecs_r.data(), file, "svdVecs_r" );
    svdVecs_i.resize(L, svdNumber); readFile(svdVecs_i.size(), svdVecs_i.data(), file, "svdVecs_i" );
    //
    svdBg_r.resize(svdNumber); readFile(svdBg_r.size(), svdBg_r.data(), file, "svdBg_r" );
    svdBg_i.resize(svdNumber); readFile(svdBg_i.size(), svdBg_i.data(), file, "svdBg_i" );

    file.close();

    K.resize(L, L);svdVecs.resize(L, svdNumber);svdBg.resize(svdNumber); 
    for(size_t i=1-1; i<=L-1; i++){
    for(size_t j=1-1; j<=L-1; j++){
        K(i,j)=K_r(i,j)+complex<double>(0.0,1.0)*K_i(i,j);
    } 
    } 
    //
    //get full svd_sym from input
    for(size_t k=1-1; k<=svdNumber-1; k++){
        for(size_t i=1-1; i<=L-1; i++){
            svdVecs(i,k)=svdVecs_r(i,k)+complex<double>(0.0,1.0)*svdVecs_i(i,k);
        }
    }
    //
    for(size_t k=1-1; k<=svdNumber-1; k++){
        svdBg(k)=svdBg_r(k)+complex<double>(0.0,1.0)*svdBg_i(k);
    }
    initialBg=svdBg;

    KpEigenStatus = 0;
    Kp.resize(0,0);
    KpEigenValue.resize( static_cast<size_t>(0) );
    KpEigenVector.resize( 0, 0 );
}

void GeneralHamiltonian_sym_icf::read_conj(const string &filename)
{
    H5File file(filename, H5F_ACC_RDONLY);
    // readFile(k_l_x, file, "k_l_x");
    // readFile(k_l_y, file, "k_l_y");
    readFile(L, file, "L");
    readFile(Nup, file, "Nup");
    readFile(Ndn, file, "Ndn");
    readFile(N, file, "N");
    readFile(svdNumber, file, "svdNumber");

    TensorHao<double,2> K_r,K_i;
    TensorHao<double,2> svdVecs_r,svdVecs_i;
    TensorHao<double,1> svdBg_r,svdBg_i;
  
    K_r.resize(L, L); readFile( K_r.size(),  K_r.data(),  file, "K_r" );
    K_i.resize(L, L); readFile( K_i.size(),  K_i.data(),  file, "K_i" );
    //Vq is not depends on k
    svdVecs_r.resize(L, svdNumber); readFile(svdVecs_r.size(), svdVecs_r.data(), file, "svdVecs_r" );
    svdVecs_i.resize(L, svdNumber); readFile(svdVecs_i.size(), svdVecs_i.data(), file, "svdVecs_i" );
    //
    svdBg_r.resize(svdNumber); readFile(svdBg_r.size(), svdBg_r.data(), file, "svdBg_r" );
    svdBg_i.resize(svdNumber); readFile(svdBg_i.size(), svdBg_i.data(), file, "svdBg_i" );

    file.close();

    K.resize(L, L);svdVecs.resize(L, svdNumber);svdBg.resize(svdNumber); 
    for(size_t i=1-1; i<=L-1; i++){
    for(size_t j=1-1; j<=L-1; j++){
        K(i,j)=K_r(j,i)+complex<double>(0.0,-1.0)*K_i(j,i);
    } 
    } 
    //
    //get full svd_sym from input
    for(size_t k=1-1; k<=svdNumber-1; k++){
        for(size_t i=1-1; i<=L-1; i++){
            svdVecs(i,k)=svdVecs_r(i,svdNumber-1-k)+complex<double>(0.0,-1.0)*svdVecs_i(i,svdNumber-1-k);
        }
    }
    //
    for(size_t k=1-1; k<=svdNumber-1; k++){
        svdBg(k)=svdBg_r(svdNumber-1-k)+complex<double>(0.0,-1.0)*svdBg_i(svdNumber-1-k);
    }
    initialBg=svdBg;

    KpEigenStatus = 0;
    Kp.resize(0,0);
    KpEigenValue.resize( static_cast<size_t>(0) );
    KpEigenVector.resize( 0, 0 );
}

void GeneralHamiltonian_sym_icf::write(const string &filename) const
{
    //
    tensor_hao::TensorHao<double,2> K_rTemp,K_iTemp;
    K_rTemp.resize(L,L);K_iTemp.resize(L,L);
    for(size_t i=1-1; i<=L-1; i++){
    for(size_t j=1-1; j<=L-1; j++){
        K_rTemp(i,j)=real(K(i,j));
        K_iTemp(i,j)=imag(K(i,j));
    }
    }
    //
    tensor_hao::TensorHao<double,2> svdVecs_rTemp,svdVecs_iTemp;
    svdVecs_rTemp.resize(L,svdNumber);svdVecs_iTemp.resize(L,svdNumber);
    for(size_t i=1-1; i<=L-1; i++){
    for(size_t kk=1-1; kk<=svdNumber-1; kk++){
        svdVecs_rTemp(i,kk)=real(svdVecs(i,kk));
        svdVecs_iTemp(i,kk)=imag(svdVecs(i,kk));
    }
    }
    //
    tensor_hao::TensorHao<double,1> svdBg_rTemp,svdBg_iTemp;
    svdBg_rTemp.resize(svdNumber);svdBg_iTemp.resize(svdNumber);
    for(size_t kk=1-1; kk<=svdNumber-1; kk++){
        svdBg_rTemp(kk)=real(svdBg(kk));
        svdBg_iTemp(kk)=imag(svdBg(kk));
    }

    H5File file(filename, H5F_ACC_TRUNC);

    // writeFile( k_l_x, file, "k_l_x" );
    // writeFile( k_l_y, file, "k_l_y" );
    writeFile( L, file, "L" );
    writeFile( Nup, file, "Nup" );
    writeFile( Ndn, file, "Ndn" );
    writeFile( N, file, "N" );
    writeFile( svdNumber, file, "svdNumber" );
    writeFile( K.size(),  K_rTemp.data(),  file, "K_r" );
    writeFile( K.size(),  K_iTemp.data(),  file, "K_i" );
    writeFile( svdVecs.size(), svdVecs_rTemp.data(), file, "svdVecs_r" );
    writeFile( svdVecs.size(), svdVecs_iTemp.data(), file, "svdVecs_i" );
    writeFile( svdBg.size(), svdBg_rTemp.data(), file, "svdBg_r" );
    writeFile( svdBg.size(), svdBg_iTemp.data(), file, "svdBg_i" );

    file.close();
}

#ifdef MPI_HAO
void MPIBcast(GeneralHamiltonian_sym_icf &buffer, int root, MPI_Comm const &comm)
{
    // MPIBcast(buffer.k_l_x, root, comm);
    // MPIBcast(buffer.k_l_y, root, comm);
    MPIBcast(buffer.L, root, comm);
    MPIBcast(buffer.Nup, root, comm);
    MPIBcast(buffer.Ndn, root, comm);
    MPIBcast(buffer.N, root, comm);
    MPIBcast(buffer.svdNumber, root, comm);
    MPIBcast(buffer.K, root, comm);
    MPIBcast(buffer.svdVecs, root, comm);
    MPIBcast(buffer.svdBg, root, comm);
    MPIBcast(buffer.initialBg, root, comm);
    MPIBcast(buffer.KpEigenStatus, root, comm);
    MPIBcast(buffer.Kp, root, comm);
    MPIBcast(buffer.KpEigenValue, root, comm);
    MPIBcast(buffer.KpEigenVector, root, comm);
}
#endif

void GeneralHamiltonian_sym_icf::writeBackGround(const string &filename) const
{
    H5File file(filename, H5F_ACC_RDWR);
    writeFile( svdBg.size(), svdBg.data(), file, "svdBg" );
    file.close();
}

void GeneralHamiltonian_sym_icf::updateBackGround(const TensorHao<complex<double>, 1> &background)
{
    if( background.size() != svdNumber ) {cout<<"Error!!! Background size is not svdNumber!"<<endl; exit(1);}
    KpEigenStatus = 0;
    svdBg = background;
}

void GeneralHamiltonian_sym_icf::updateBackGround(TensorHao<complex<double>, 1> &&background)
{
    if( background.size() != svdNumber ) {cout<<"Error!!! Background size is not svdNumber!"<<endl; exit(1);}
    KpEigenStatus = 0;
    svdBg = move(background);
}

Hop GeneralHamiltonian_sym_icf::returnExpMinusAlphaK(double alpha)
{
    setKpEigenValueAndVector();

    TensorHao<complex<double>,2> matrix(L,L);
    BL_NAME(gmm)( KpEigenVector, dMultiMatrix( exp(-alpha*KpEigenValue), conj(trans(KpEigenVector) )), matrix );

    Hop hop(L);
    complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
    hop.logw = alpha*0.5*bg2;
    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < L; ++j) hop.matrix(j,i) = matrix(j,i);
    }
    return hop;
}

Hop2s GeneralHamiltonian_sym_icf::returnExpMinusAlphaK2s(double alpha)
{
    setKpEigenValueAndVector();
    if( 2*(L/2) != L){
        cout<<"Error: 2s is not allowed in GeneralHamiltonian_sym_icf"<<endl;
        exit(1);
    }
    TensorHao<complex<double>,2> matrix(L,L);
    TensorHao<complex<double>,2> matrixUp(L/2,L/2);
    TensorHao<complex<double>,2> matrixDn(L/2,L/2);
    BL_NAME(gmm)( KpEigenVector, dMultiMatrix( exp(-alpha*KpEigenValue), conj(trans(KpEigenVector) )), matrix );

    Hop2s hop2s(L/2);
    complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
    hop2s.logw = alpha*0.5*bg2;
    for(size_t i = 0; i < L/2; ++i)
    {
        for(size_t j = 0; j < L/2; ++j) hop2s.matrixUp(j,i) = matrix(j,i);
    }
    for(size_t i = 0; i < L/2; ++i)
    {
        for(size_t j = 0; j < L/2; ++j) hop2s.matrixDn(j,i) = matrix(j+L/2,i+L/2);
    }
    return hop2s;
}

Hop2s GeneralHamiltonian_sym_icf::returnExpMinusAlphaK2s_nonHermitian(double alpha, std::string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder)
{
    //generate Hop from LogHop for nonHermitian Kp
    LogHop logHop=returnLogExpMinusAlphaK(alpha);
    //Aux walker to store nonHermitian expm
    SD walkerIdentity(L,L);
    TensorHao<complex<double>,2> identity(L,L); identity=0.0;
    for(size_t i = 0; i < L; ++i)
    {
        identity(i,i) = 1.0; 
    }
    walkerIdentity.wfRef()=identity;
    SD walkerTemp=walkerIdentity;
    //
    if( 2*(L/2) != L){
        cout<<"Error: 2s is not allowed in GeneralHamiltonian_sym_icf"<<endl;
        exit(1);
    }
    Hop2s hop2s(L/2);
    //
    LogHopSDOperation_icf oneBodyWalkerOperation;
    oneBodyWalkerOperation.reset(flag, taylorOrder, accuracy, baseTaylorOrder);
    oneBodyWalkerOperation.applyToRight(logHop, walkerIdentity, walkerTemp);
    //
    for(size_t i = 0; i < L/2; ++i)
    {
        for(size_t j = 0; j < L/2; ++j) hop2s.matrixUp(i,j) = walkerTemp.getWf()(i,j);
    }
    for(size_t i = 0; i < L/2; ++i)
    {
        for(size_t j = 0; j < L/2; ++j) hop2s.matrixDn(i,j) = walkerTemp.getWf()(i+L/2,j+L/2);
    }
    hop2s.logw = walkerTemp.getLogw();
    //
    return hop2s;
}

Hop GeneralHamiltonian_sym_icf::returnExpMinusAlphaK_nonHermitian(double alpha, std::string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder)
{
    //generate Hop from LogHop for nonHermitian Kp
    LogHop logHop=returnLogExpMinusAlphaK(alpha);
    //Aux walker to store nonHermitian expm
    SD walkerIdentity(L,L);
    TensorHao<complex<double>,2> identity(L,L); identity=0.0;
    for(size_t i = 0; i < L; ++i)
    {
        identity(i,i) = 1.0; 
    }
    walkerIdentity.wfRef()=identity;
    SD walkerTemp=walkerIdentity;
    //
    Hop hop(L);
    //
    LogHopSDOperation_icf oneBodyWalkerOperation;
    oneBodyWalkerOperation.reset(flag, taylorOrder, accuracy, baseTaylorOrder);
    oneBodyWalkerOperation.applyToRight(logHop, walkerIdentity, walkerTemp);
    //
    hop.matrix = walkerTemp.getWf();
    hop.logw = walkerTemp.getLogw();
    //
    return hop;
}

LogHop GeneralHamiltonian_sym_icf::returnLogExpMinusAlphaK(double alpha)
{
    setKp();

    LogHop logHop(L);
    complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
    logHop.logw = alpha*0.5*bg2;
    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < L; ++j) logHop.matrix(j,i) = -alpha*Kp(j,i);
    }
    return logHop;
}

LogHop2s GeneralHamiltonian_sym_icf::returnLogExpMinusAlphaK2s(double alpha)
{
    setKp();

    LogHop2s logHop2s(L/2);
    complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
    logHop2s.logw = alpha*0.5*bg2;
    for(size_t i = 0; i < L/2; ++i)
    {
        for(size_t j = 0; j < L/2; ++j) logHop2s.matrixUp(j,i) = -alpha*Kp(j,i);
    }
    for(size_t i = 0; i < L/2; ++i)
    {
        for(size_t j = 0; j < L/2; ++j) logHop2s.matrixDn(j,i) = -alpha*Kp(j+L/2,i+L/2);
    }
    return logHop2s;
}


SVD_sym GeneralHamiltonian_sym_icf::returnExpMinusAlphaV(double alpha)
{
    return SVD_sym(alpha, svdVecs, svdBg);
}

SVD_sym GeneralHamiltonian_sym_icf::returnExpMinusAlphaV_daggerSqrtDt(double alpha)
{
    return SVD_sym(alpha, svdVecs, svdBg, -1);;//ATTENTION: extra -1 here for sqrt(-1*dt) in SVD_sym
}


void GeneralHamiltonian_sym_icf::setKp()
{
    if( KpEigenStatus >=1 ) return;

    Kp = K;
    // TensorHaoRef<complex<double>,2> vecs(L2, svdNumber); vecs.point( svdVecs.data() );
    // TensorHaoRef<complex<double>,1> vecsBg(L2); vecsBg.point( Kp.data() );
    // BL_NAME(gemv)(vecs, svdBg, vecsBg, 'N', 1.0, 1.0);

    TensorHao<complex<double>,2> matrix_temp(L,L); matrix_temp=0.0;
    for(int i=1-1; i<=svdNumber-1; i++){
        for(int j=1-1; j<=L-1; j++){
            matrix_temp(j,j) += svdVecs(j,i) * svdBg(i);
        }
    }
    Kp = Kp + matrix_temp;
    
    KpEigenStatus=1;
}

void GeneralHamiltonian_sym_icf::setKpEigenValueAndVector()
{
    if( KpEigenStatus >=2 ) return;

    setKp();
    checkHermitian(Kp, 1e-8);
    KpEigenVector = Kp;
    KpEigenValue.resize(L);
    BL_NAME(eigen)(KpEigenVector, KpEigenValue);

    KpEigenStatus = 2;
}

double GeneralHamiltonian_sym_icf::getMemory() const
{
    double mem(0.0);

    mem += 8.0*4;
    mem += K.getMemory();
    mem += svdVecs.getMemory();
    mem += svdBg.getMemory();
    mem += initialBg.getMemory();

    mem += 8.0;
    mem += Kp.getMemory();
    mem += KpEigenValue.getMemory();
    mem += KpEigenVector.getMemory();

    return mem;
}

GeneralHamiltonian_sym_icf::GeneralHamiltonian_sym_icf(const GeneralHamiltonian_sym_icf &x)  { }

GeneralHamiltonian_sym_icf &GeneralHamiltonian_sym_icf::operator=(const GeneralHamiltonian_sym_icf &x) { return *this; }