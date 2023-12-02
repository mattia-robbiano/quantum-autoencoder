from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
#from qiskit.utils import algorithm_globals

def AnsatzBuilder(QubitNumber, Depth):
    return RealAmplitudes(QubitNumber, reps=Depth)

def SwaptestBuilder(TrashSpaceDimension):
    QubitNumber = 2*TrashSpaceDimension + 1
    QuantReg = QuantumRegister(QubitNumber,"q")
    ClassReg = ClassicalRegister(1,"c")
    SwaptestCircuit = QuantumCircuit(QuantReg, ClassReg)
    AuxiliaryQubitLable = QubitNumber-1
    
    #INITIALIZE REFERENCE QUBITS TO 0
    for i in range(TrashSpaceDimension,2*TrashSpaceDimension): SwaptestCircuit.reset(i)

    #H GATE ON AUXILIARY QUBIT
    SwaptestCircuit.h(AuxiliaryQubitLable)
    #CSWAP GATES BETWEEN TRASH AND REFERENCE QUBITS TARGETING AUXILIARY QUBIT
    for i in range(TrashSpaceDimension):
        SwaptestCircuit.cswap(AuxiliaryQubitLable, i, TrashSpaceDimension+i)
    #H GATE ON AUXILIARY QUBIT
    SwaptestCircuit.h(AuxiliaryQubitLable)
    SwaptestCircuit.measure(AuxiliaryQubitLable, ClassReg[0])

    return SwaptestCircuit

def EncoderBuilder(InputStateDimension, EncodedStateDimension, Depth):
    #Calculate dimension of spaces
    LatentSpaceDimension = EncodedStateDimension
    TrashSpaceDimension = InputStateDimension - EncodedStateDimension
    ReferenceSpaceDimension = TrashSpaceDimension
    TotalQubitNumber = LatentSpaceDimension + TrashSpaceDimension + ReferenceSpaceDimension +1 #+1 (for auxiliary qubit)

    #Initialize circuit
    QuantReg = QuantumRegister(TotalQubitNumber,"q")
    ClassReg = ClassicalRegister(1,"c")
    Circuit = QuantumCircuit(QuantReg, ClassReg)

    #Add AnsatzCircuit
    Ansatz = AnsatzBuilder(LatentSpaceDimension+TrashSpaceDimension, Depth)
    Circuit.compose(Ansatz,range(0, InputStateDimension), inplace=True)
    #Barrier to define optimization regions
    Circuit.barrier()
    #Add SwapTestCircuit
    Circuit.compose(SwaptestBuilder(TrashSpaceDimension),range(LatentSpaceDimension, TotalQubitNumber), inplace=True)
    
    return Circuit