from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

#Funzione che prende in input il numero di qubit e la profondità del circuito e restituisce il circuito ansatz
#patametrico di tipo RealAmplitudes corrispondente.
def AnsatzBuilder(QubitNumber, Depth):
    return RealAmplitudes(QubitNumber, reps=Depth)

#Funzione che prende in input il numero di qubit nel TrashSpace e restituisce il circuito SwapTest corrispondente.
def SwaptestBuilder(TrashSpaceDimension):
    QubitNumber = 2*TrashSpaceDimension + 1 #calcolo il numero di qubit totali necessari
    #Inizializzo i registri quantistici e classici
    QuantReg = QuantumRegister(QubitNumber,"q")
    ClassReg = ClassicalRegister(1,"c")
    #Inizializzo il circuito SwapTest
    SwaptestCircuit = QuantumCircuit(QuantReg, ClassReg)
    AuxiliaryQubitLable = QubitNumber-1 #Label dell'ultimo qubit (quello ausiliario)
    
    #Inizializzo i qubit nel ReferenceSpace a 0
    for i in range(TrashSpaceDimension,2*TrashSpaceDimension): SwaptestCircuit.reset(i)

    #Costruisco il circuito SwapTest

    SwaptestCircuit.h(AuxiliaryQubitLable) #porta H sul qubit ausiliario
    #CSWAP gates tra Trash e Reference qubits targeting Auxiliary qubit
    for i in range(TrashSpaceDimension):
        SwaptestCircuit.cswap(AuxiliaryQubitLable, i, TrashSpaceDimension+i)

    SwaptestCircuit.h(AuxiliaryQubitLable) #porta H sul qubit ausiliario
    SwaptestCircuit.measure(AuxiliaryQubitLable, ClassReg[0]) #misura il qubit ausiliario

    return SwaptestCircuit

#Funzione che prende in input la dimensione dello spazio di input, la dimensione dello spazio di output e la profondità
#del circuito ansatz e restituisce il circuito Encoder completo (usando SwaptestBuilder e AnsatzBuilder)
def EncoderBuilder(InputStateDimension, EncodedStateDimension, Depth):
    #Calcolo le dimensioni dei vari spazi
    LatentSpaceDimension = EncodedStateDimension
    TrashSpaceDimension = InputStateDimension - EncodedStateDimension
    ReferenceSpaceDimension = TrashSpaceDimension
    TotalQubitNumber = LatentSpaceDimension + TrashSpaceDimension + ReferenceSpaceDimension +1 #+1 (for auxiliary qubit)

    #Inizializzo i registri quantistici e classici
    QuantReg = QuantumRegister(TotalQubitNumber,"q")
    ClassReg = ClassicalRegister(1,"c")
    Circuit = QuantumCircuit(QuantReg, ClassReg)

    #Aggiungo AnsatzCircuit
    Ansatz = AnsatzBuilder(LatentSpaceDimension+TrashSpaceDimension, Depth)
    Circuit.compose(Ansatz,range(0, InputStateDimension), inplace=True)
    #Barriera per delimitare i vari spazi
    Circuit.barrier()
    #Aggiungo SwapTestCircuit
    Circuit.compose(SwaptestBuilder(TrashSpaceDimension),range(LatentSpaceDimension, TotalQubitNumber), inplace=True)
    
    return Circuit