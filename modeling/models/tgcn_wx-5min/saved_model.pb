á&
ô
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.12v2.4.0-49-g85c8b2a817f8¬"
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
°
*fixed_adjacency_graph_convolution_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*;
shared_name,*fixed_adjacency_graph_convolution_3/kernel
©
>fixed_adjacency_graph_convolution_3/kernel/Read/ReadVariableOpReadVariableOp*fixed_adjacency_graph_convolution_3/kernel*
_output_shapes

:
*
dtype0
¬
(fixed_adjacency_graph_convolution_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*9
shared_name*(fixed_adjacency_graph_convolution_3/bias
¥
<fixed_adjacency_graph_convolution_3/bias/Read/ReadVariableOpReadVariableOp(fixed_adjacency_graph_convolution_3/bias*
_output_shapes

:F*
dtype0

lstm_3/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F **
shared_namelstm_3/lstm_cell_3/kernel

-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/kernel*
_output_shapes
:	F *
dtype0
¤
#lstm_3/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È *4
shared_name%#lstm_3/lstm_cell_3/recurrent_kernel

7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_3/recurrent_kernel* 
_output_shapes
:
È *
dtype0

lstm_3/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namelstm_3/lstm_cell_3/bias

+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/bias*
_output_shapes	
: *
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ÈF*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	ÈF*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:F*
dtype0
¦
%fixed_adjacency_graph_convolution_3/AVarHandleOp*
_output_shapes
: *
dtype0*
shape
:FF*6
shared_name'%fixed_adjacency_graph_convolution_3/A

9fixed_adjacency_graph_convolution_3/A/Read/ReadVariableOpReadVariableOp%fixed_adjacency_graph_convolution_3/A*
_output_shapes

:FF*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
¾
1Adam/fixed_adjacency_graph_convolution_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*B
shared_name31Adam/fixed_adjacency_graph_convolution_3/kernel/m
·
EAdam/fixed_adjacency_graph_convolution_3/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_3/kernel/m*
_output_shapes

:
*
dtype0
º
/Adam/fixed_adjacency_graph_convolution_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_3/bias/m
³
CAdam/fixed_adjacency_graph_convolution_3/bias/m/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_3/bias/m*
_output_shapes

:F*
dtype0

 Adam/lstm_3/lstm_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F *1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/m

4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/m*
_output_shapes
:	F *
dtype0
²
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È *;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
«
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m* 
_output_shapes
:
È *
dtype0

Adam/lstm_3/lstm_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/lstm_3/lstm_cell_3/bias/m

2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/m*
_output_shapes	
: *
dtype0

Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ÈF*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	ÈF*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:F*
dtype0

Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
¾
1Adam/fixed_adjacency_graph_convolution_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*B
shared_name31Adam/fixed_adjacency_graph_convolution_3/kernel/v
·
EAdam/fixed_adjacency_graph_convolution_3/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_3/kernel/v*
_output_shapes

:
*
dtype0
º
/Adam/fixed_adjacency_graph_convolution_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_3/bias/v
³
CAdam/fixed_adjacency_graph_convolution_3/bias/v/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_3/bias/v*
_output_shapes

:F*
dtype0

 Adam/lstm_3/lstm_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F *1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/v

4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/v*
_output_shapes
:	F *
dtype0
²
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È *;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
«
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v* 
_output_shapes
:
È *
dtype0

Adam/lstm_3/lstm_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/lstm_3/lstm_cell_3/bias/v

2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/v*
_output_shapes	
: *
dtype0

Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ÈF*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	ÈF*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:F*
dtype0

NoOpNoOp
þH
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¹H
value¯HB¬H B¥H
Ù
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
¢
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
layer-8
layer_with_weights-2
layer-9
trainable_variables
 regularization_losses
!	variables
"	keras_api
ô
#iter

$beta_1

%beta_2
	&decay
'learning_ratem¤m¥(m¦)m§*m¨+m©,mª-m«.m¬v­v®(v¯)v°*v±+v²,v³-v´.vµ
?
0
1
(2
)3
*4
+5
,6
-7
.8
 
F
0
1
(2
)3
/4
*5
+6
,7
-8
.9
­
0metrics

1layers
trainable_variables
regularization_losses
2layer_regularization_losses
3layer_metrics
4non_trainable_variables
	variables
 
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

5layers
6metrics
trainable_variables
regularization_losses
7layer_regularization_losses
8layer_metrics
9non_trainable_variables
	variables
 
 
 
­

:layers
;metrics
trainable_variables
regularization_losses
<layer_regularization_losses
=layer_metrics
>non_trainable_variables
	variables
 

?	keras_api
R
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
o
/A

(kernel
)bias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
R
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
R
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
R
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
l
Tcell
U
state_spec
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
R
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
h

-kernel
.bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
1
(0
)1
*2
+3
,4
-5
.6
 
8
(0
)1
/2
*3
+4
,5
-6
.7
­
bmetrics

clayers
trainable_variables
 regularization_losses
dlayer_regularization_losses
elayer_metrics
fnon_trainable_variables
!	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE*fixed_adjacency_graph_convolution_3/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE(fixed_adjacency_graph_convolution_3/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_3/lstm_cell_3/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_3/lstm_cell_3/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_3/lstm_cell_3/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_7/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_7/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%fixed_adjacency_graph_convolution_3/A&variables/4/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

0
1
2
3
 
 

/0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
­

ilayers
jmetrics
@trainable_variables
Aregularization_losses
klayer_regularization_losses
llayer_metrics
mnon_trainable_variables
B	variables

(0
)1
 

(0
)1
/2
­

nlayers
ometrics
Dtrainable_variables
Eregularization_losses
player_regularization_losses
qlayer_metrics
rnon_trainable_variables
F	variables
 
 
 
­

slayers
tmetrics
Htrainable_variables
Iregularization_losses
ulayer_regularization_losses
vlayer_metrics
wnon_trainable_variables
J	variables
 
 
 
­

xlayers
ymetrics
Ltrainable_variables
Mregularization_losses
zlayer_regularization_losses
{layer_metrics
|non_trainable_variables
N	variables
 
 
 
¯

}layers
~metrics
Ptrainable_variables
Qregularization_losses
layer_regularization_losses
layer_metrics
non_trainable_variables
R	variables


*kernel
+recurrent_kernel
,bias
trainable_variables
regularization_losses
	variables
	keras_api
 

*0
+1
,2
 

*0
+1
,2
¿
metrics
layers
Vtrainable_variables
Wregularization_losses
 layer_regularization_losses
layer_metrics
non_trainable_variables
states
X	variables
 
 
 
²
layers
metrics
Ztrainable_variables
[regularization_losses
 layer_regularization_losses
layer_metrics
non_trainable_variables
\	variables

-0
.1
 

-0
.1
²
layers
metrics
^trainable_variables
_regularization_losses
 layer_regularization_losses
layer_metrics
non_trainable_variables
`	variables
 
F
0
1
2
3
4
5
6
7
8
9
 
 

/0
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
 
 
 
 
 
 
 
 
 

/0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

*0
+1
,2
 

*0
+1
,2
µ
layers
 metrics
trainable_variables
regularization_losses
 ¡layer_regularization_losses
¢layer_metrics
£non_trainable_variables
	variables
 

T0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
 
 
 
 
 
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_3/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_3/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_7/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_7/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_3/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_3/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_7/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_7/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_11Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿF
Ø
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11dense_8/kerneldense_8/bias%fixed_adjacency_graph_convolution_3/A*fixed_adjacency_graph_convolution_3/kernel(fixed_adjacency_graph_convolution_3/biaslstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_60178
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
î
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp>fixed_adjacency_graph_convolution_3/kernel/Read/ReadVariableOp<fixed_adjacency_graph_convolution_3/bias/Read/ReadVariableOp-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_3/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp9fixed_adjacency_graph_convolution_3/A/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_3/kernel/m/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_3/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_3/kernel/v/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_3/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_62399


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate*fixed_adjacency_graph_convolution_3/kernel(fixed_adjacency_graph_convolution_3/biaslstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biasdense_7/kerneldense_7/bias%fixed_adjacency_graph_convolution_3/Atotalcounttotal_1count_1Adam/dense_8/kernel/mAdam/dense_8/bias/m1Adam/fixed_adjacency_graph_convolution_3/kernel/m/Adam/fixed_adjacency_graph_convolution_3/bias/m Adam/lstm_3/lstm_cell_3/kernel/m*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mAdam/lstm_3/lstm_cell_3/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/v1Adam/fixed_adjacency_graph_convolution_3/kernel/v/Adam/fixed_adjacency_graph_convolution_3/bias/v Adam/lstm_3/lstm_cell_3/kernel/v*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vAdam/lstm_3/lstm_cell_3/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_62520©ó 


&__inference_lstm_3_layer_call_fn_61791

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_595102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
J
Ô	
lstm_3_while_body_61230*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0?
;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0>
:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor;
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource=
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource<
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource¢/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp¢.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp¢0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpÑ
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype020
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpð
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lstm_3/while/lstm_cell_3/MatMulâ
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype022
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpÙ
!lstm_3/while/lstm_cell_3/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lstm_3/while/lstm_cell_3/MatMul_1Ð
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/MatMul:product:0+lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/while/lstm_cell_3/addÚ
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype021
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpÝ
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd lstm_3/while/lstm_cell_3/add:z:07lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 lstm_3/while/lstm_cell_3/BiasAdd
lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_3/while/lstm_cell_3/Const
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim§
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:0)lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2 
lstm_3/while/lstm_cell_3/split«
 lstm_3/while/lstm_cell_3/SigmoidSigmoid'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 lstm_3/while/lstm_cell_3/Sigmoid¯
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm_3/while/lstm_cell_3/Sigmoid_1º
lstm_3/while/lstm_cell_3/mulMul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/while/lstm_cell_3/mulÉ
lstm_3/while/lstm_cell_3/mul_1Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_3/while/lstm_cell_3/mul_1Â
lstm_3/while/lstm_cell_3/add_1AddV2 lstm_3/while/lstm_cell_3/mul:z:0"lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_3/while/lstm_cell_3/add_1¯
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm_3/while/lstm_cell_3/Sigmoid_2Æ
lstm_3/while/lstm_cell_3/mul_2Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_3/while/lstm_cell_3/mul_2
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/y
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/y
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity£
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations0^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1
lstm_3/while/Identity_2Identitylstm_3/while/add:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2¸
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3«
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_2:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/while/Identity_4«
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_1:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/while/Identity_5"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"Ä
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2b
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
Þ
`
D__inference_permute_3_layer_call_and_return_conditional_losses_58606

inputs
identityy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transpose
IdentityIdentitytranspose:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
Ù
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_58714

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2©
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity­

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1­

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates


&__inference_lstm_3_layer_call_fn_62111
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_590772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0
õ$

while_body_59140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_3_59164_0
while_lstm_cell_3_59166_0
while_lstm_cell_3_59168_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_3_59164
while_lstm_cell_3_59166
while_lstm_cell_3_59168¢)while/lstm_cell_3/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_59164_0while_lstm_cell_3_59166_0while_lstm_cell_3_59168_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_587142+
)while/lstm_cell_3/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1*^while/lstm_cell_3/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4Ã
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2*^while/lstm_cell_3/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_59164while_lstm_cell_3_59164_0"4
while_lstm_cell_3_59166while_lstm_cell_3_59166_0"4
while_lstm_cell_3_59168while_lstm_cell_3_59168_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
Ý
¡
!__inference__traced_restore_62520
file_prefix#
assignvariableop_dense_8_kernel#
assignvariableop_1_dense_8_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rateA
=assignvariableop_7_fixed_adjacency_graph_convolution_3_kernel?
;assignvariableop_8_fixed_adjacency_graph_convolution_3_bias0
,assignvariableop_9_lstm_3_lstm_cell_3_kernel;
7assignvariableop_10_lstm_3_lstm_cell_3_recurrent_kernel/
+assignvariableop_11_lstm_3_lstm_cell_3_bias&
"assignvariableop_12_dense_7_kernel$
 assignvariableop_13_dense_7_bias=
9assignvariableop_14_fixed_adjacency_graph_convolution_3_a
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1-
)assignvariableop_19_adam_dense_8_kernel_m+
'assignvariableop_20_adam_dense_8_bias_mI
Eassignvariableop_21_adam_fixed_adjacency_graph_convolution_3_kernel_mG
Cassignvariableop_22_adam_fixed_adjacency_graph_convolution_3_bias_m8
4assignvariableop_23_adam_lstm_3_lstm_cell_3_kernel_mB
>assignvariableop_24_adam_lstm_3_lstm_cell_3_recurrent_kernel_m6
2assignvariableop_25_adam_lstm_3_lstm_cell_3_bias_m-
)assignvariableop_26_adam_dense_7_kernel_m+
'assignvariableop_27_adam_dense_7_bias_m-
)assignvariableop_28_adam_dense_8_kernel_v+
'assignvariableop_29_adam_dense_8_bias_vI
Eassignvariableop_30_adam_fixed_adjacency_graph_convolution_3_kernel_vG
Cassignvariableop_31_adam_fixed_adjacency_graph_convolution_3_bias_v8
4assignvariableop_32_adam_lstm_3_lstm_cell_3_kernel_vB
>assignvariableop_33_adam_lstm_3_lstm_cell_3_recurrent_kernel_v6
2assignvariableop_34_adam_lstm_3_lstm_cell_3_bias_v-
)assignvariableop_35_adam_dense_7_kernel_v+
'assignvariableop_36_adam_dense_7_bias_v
identity_38¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9è
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ô
valueêBç&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÚ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Â
AssignVariableOp_7AssignVariableOp=assignvariableop_7_fixed_adjacency_graph_convolution_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8À
AssignVariableOp_8AssignVariableOp;assignvariableop_8_fixed_adjacency_graph_convolution_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9±
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_3_lstm_cell_3_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¿
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_3_lstm_cell_3_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11³
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_3_lstm_cell_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ª
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¨
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Á
AssignVariableOp_14AssignVariableOp9assignvariableop_14_fixed_adjacency_graph_convolution_3_aIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¡
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18£
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19±
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_8_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¯
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_8_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Í
AssignVariableOp_21AssignVariableOpEassignvariableop_21_adam_fixed_adjacency_graph_convolution_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ë
AssignVariableOp_22AssignVariableOpCassignvariableop_22_adam_fixed_adjacency_graph_convolution_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¼
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_3_lstm_cell_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Æ
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_lstm_3_lstm_cell_3_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25º
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_lstm_3_lstm_cell_3_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26±
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_7_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¯
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_7_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_8_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¯
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_8_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Í
AssignVariableOp_30AssignVariableOpEassignvariableop_30_adam_fixed_adjacency_graph_convolution_3_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ë
AssignVariableOp_31AssignVariableOpCassignvariableop_31_adam_fixed_adjacency_graph_convolution_3_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¼
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_3_lstm_cell_3_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Æ
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_lstm_3_lstm_cell_3_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34º
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_lstm_3_lstm_cell_3_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35±
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_7_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¯
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_7_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37ÿ
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*«
_input_shapes
: :::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


&__inference_lstm_3_layer_call_fn_62122
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_592092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_62134

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¹R
±
__inference__traced_save_62399
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopI
Esavev2_fixed_adjacency_graph_convolution_3_kernel_read_readvariableopG
Csavev2_fixed_adjacency_graph_convolution_3_bias_read_readvariableop8
4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableopD
@savev2_fixed_adjacency_graph_convolution_3_a_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_3_kernel_m_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_3_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_3_kernel_v_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_3_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameâ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ô
valueêBç&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÔ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopEsavev2_fixed_adjacency_graph_convolution_3_kernel_read_readvariableopCsavev2_fixed_adjacency_graph_convolution_3_bias_read_readvariableop4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop@savev2_fixed_adjacency_graph_convolution_3_a_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_3_kernel_m_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_3_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_3_kernel_v_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_3_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*®
_input_shapes
: ::: : : : : :
:F:	F :
È : :	ÈF:F:FF: : : : :::
:F:	F :
È : :	ÈF:F:::
:F:	F :
È : :	ÈF:F: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
:$	 

_output_shapes

:F:%
!

_output_shapes
:	F :&"
 
_output_shapes
:
È :!

_output_shapes	
: :%!

_output_shapes
:	ÈF: 

_output_shapes
:F:$ 

_output_shapes

:FF:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
:$ 

_output_shapes

:F:%!

_output_shapes
:	F :&"
 
_output_shapes
:
È :!

_output_shapes	
: :%!

_output_shapes
:	ÈF: 

_output_shapes
:F:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
:$  

_output_shapes

:F:%!!

_output_shapes
:	F :&""
 
_output_shapes
:
È :!#

_output_shapes	
: :%$!

_output_shapes
:	ÈF: %

_output_shapes
:F:&

_output_shapes
: 
õ$

while_body_59008
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_3_59032_0
while_lstm_cell_3_59034_0
while_lstm_cell_3_59036_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_3_59032
while_lstm_cell_3_59034
while_lstm_cell_3_59036¢)while/lstm_cell_3/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_59032_0while_lstm_cell_3_59034_0while_lstm_cell_3_59036_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_586832+
)while/lstm_cell_3/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1*^while/lstm_cell_3/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4Ã
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2*^while/lstm_cell_3/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_59032while_lstm_cell_3_59032_0"4
while_lstm_cell_3_59034while_lstm_cell_3_59034_0"4
while_lstm_cell_3_59036while_lstm_cell_3_59036_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ñ
÷
(__inference_T-GCN-WX_layer_call_fn_60143
input_11
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_601202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_11
þ
Ù
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_58683

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2©
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity­

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1­

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates
áU
Ô
model_8_lstm_3_while_body_60360:
6model_8_lstm_3_while_model_8_lstm_3_while_loop_counter@
<model_8_lstm_3_while_model_8_lstm_3_while_maximum_iterations$
 model_8_lstm_3_while_placeholder&
"model_8_lstm_3_while_placeholder_1&
"model_8_lstm_3_while_placeholder_2&
"model_8_lstm_3_while_placeholder_39
5model_8_lstm_3_while_model_8_lstm_3_strided_slice_1_0u
qmodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0E
Amodel_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0G
Cmodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0F
Bmodel_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0!
model_8_lstm_3_while_identity#
model_8_lstm_3_while_identity_1#
model_8_lstm_3_while_identity_2#
model_8_lstm_3_while_identity_3#
model_8_lstm_3_while_identity_4#
model_8_lstm_3_while_identity_57
3model_8_lstm_3_while_model_8_lstm_3_strided_slice_1s
omodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensorC
?model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceE
Amodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceD
@model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource¢7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp¢6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp¢8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpá
Fmodel_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2H
Fmodel_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0 model_8_lstm_3_while_placeholderOmodel_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02:
8model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItemó
6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOpAmodel_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype028
6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp
'model_8/lstm_3/while/lstm_cell_3/MatMulMatMul?model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0>model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'model_8/lstm_3/while/lstm_cell_3/MatMulú
8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOpCmodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02:
8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpù
)model_8/lstm_3/while/lstm_cell_3/MatMul_1MatMul"model_8_lstm_3_while_placeholder_2@model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)model_8/lstm_3/while/lstm_cell_3/MatMul_1ð
$model_8/lstm_3/while/lstm_cell_3/addAddV21model_8/lstm_3/while/lstm_cell_3/MatMul:product:03model_8/lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$model_8/lstm_3/while/lstm_cell_3/addò
7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOpBmodel_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype029
7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpý
(model_8/lstm_3/while/lstm_cell_3/BiasAddBiasAdd(model_8/lstm_3/while/lstm_cell_3/add:z:0?model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(model_8/lstm_3/while/lstm_cell_3/BiasAdd
&model_8/lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_8/lstm_3/while/lstm_cell_3/Const¦
0model_8/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_8/lstm_3/while/lstm_cell_3/split/split_dimÇ
&model_8/lstm_3/while/lstm_cell_3/splitSplit9model_8/lstm_3/while/lstm_cell_3/split/split_dim:output:01model_8/lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2(
&model_8/lstm_3/while/lstm_cell_3/splitÃ
(model_8/lstm_3/while/lstm_cell_3/SigmoidSigmoid/model_8/lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(model_8/lstm_3/while/lstm_cell_3/SigmoidÇ
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid/model_8/lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2,
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_1Ú
$model_8/lstm_3/while/lstm_cell_3/mulMul.model_8/lstm_3/while/lstm_cell_3/Sigmoid_1:y:0"model_8_lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$model_8/lstm_3/while/lstm_cell_3/mulé
&model_8/lstm_3/while/lstm_cell_3/mul_1Mul,model_8/lstm_3/while/lstm_cell_3/Sigmoid:y:0/model_8/lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&model_8/lstm_3/while/lstm_cell_3/mul_1â
&model_8/lstm_3/while/lstm_cell_3/add_1AddV2(model_8/lstm_3/while/lstm_cell_3/mul:z:0*model_8/lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&model_8/lstm_3/while/lstm_cell_3/add_1Ç
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid/model_8/lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2,
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_2æ
&model_8/lstm_3/while/lstm_cell_3/mul_2Mul.model_8/lstm_3/while/lstm_cell_3/Sigmoid_2:y:0*model_8/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&model_8/lstm_3/while/lstm_cell_3/mul_2ª
9model_8/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_8_lstm_3_while_placeholder_1 model_8_lstm_3_while_placeholder*model_8/lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9model_8/lstm_3/while/TensorArrayV2Write/TensorListSetItemz
model_8/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_8/lstm_3/while/add/y¥
model_8/lstm_3/while/addAddV2 model_8_lstm_3_while_placeholder#model_8/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/while/add~
model_8/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_8/lstm_3/while/add_1/yÁ
model_8/lstm_3/while/add_1AddV26model_8_lstm_3_while_model_8_lstm_3_while_loop_counter%model_8/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/while/add_1¹
model_8/lstm_3/while/IdentityIdentitymodel_8/lstm_3/while/add_1:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
model_8/lstm_3/while/IdentityÛ
model_8/lstm_3/while/Identity_1Identity<model_8_lstm_3_while_model_8_lstm_3_while_maximum_iterations8^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_3/while/Identity_1»
model_8/lstm_3/while/Identity_2Identitymodel_8/lstm_3/while/add:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_3/while/Identity_2è
model_8/lstm_3/while/Identity_3IdentityImodel_8/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_3/while/Identity_3Û
model_8/lstm_3/while/Identity_4Identity*model_8/lstm_3/while/lstm_cell_3/mul_2:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
model_8/lstm_3/while/Identity_4Û
model_8/lstm_3/while/Identity_5Identity*model_8/lstm_3/while/lstm_cell_3/add_1:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
model_8/lstm_3/while/Identity_5"G
model_8_lstm_3_while_identity&model_8/lstm_3/while/Identity:output:0"K
model_8_lstm_3_while_identity_1(model_8/lstm_3/while/Identity_1:output:0"K
model_8_lstm_3_while_identity_2(model_8/lstm_3/while/Identity_2:output:0"K
model_8_lstm_3_while_identity_3(model_8/lstm_3/while/Identity_3:output:0"K
model_8_lstm_3_while_identity_4(model_8/lstm_3/while/Identity_4:output:0"K
model_8_lstm_3_while_identity_5(model_8/lstm_3/while/Identity_5:output:0"
@model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resourceBmodel_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"
Amodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceCmodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"
?model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceAmodel_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"l
3model_8_lstm_3_while_model_8_lstm_3_strided_slice_15model_8_lstm_3_while_model_8_lstm_3_strided_slice_1_0"ä
omodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensorqmodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2r
7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2p
6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2t
8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ª
¾
while_cond_61696
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_61696___redundant_placeholder03
/while_while_cond_61696___redundant_placeholder13
/while_while_cond_61696___redundant_placeholder23
/while_while_cond_61696___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
­
ä
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60120

inputs
dense_8_60096
dense_8_60098
model_8_60102
model_8_60104
model_8_60106
model_8_60108
model_8_60110
model_8_60112
model_8_60114
model_8_60116
identity¢dense_8/StatefulPartitionedCall¢model_8/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_60096dense_8_60098*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_599142!
dense_8/StatefulPartitionedCallÿ
reshape_13/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_599432
reshape_13/PartitionedCall
model_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0model_8_60102model_8_60104model_8_60106model_8_60108model_8_60110model_8_60112model_8_60114model_8_60116*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598612!
model_8/StatefulPartitionedCallÀ
IdentityIdentity(model_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
§
Ø
'__inference_model_8_layer_call_fn_59829
input_12
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_12
ª
¾
while_cond_62016
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_62016___redundant_placeholder03
/while_while_cond_62016___redundant_placeholder13
/while_while_cond_62016___redundant_placeholder23
/while_while_cond_62016___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
ª
¾
while_cond_61867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_61867___redundant_placeholder03
/while_while_cond_61867___redundant_placeholder13
/while_while_cond_61867___redundant_placeholder23
/while_while_cond_61867___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
ª
¾
while_cond_59007
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_59007___redundant_placeholder03
/while_while_cond_59007___redundant_placeholder13
/while_while_cond_59007___redundant_placeholder23
/while_while_cond_59007___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
¬
F
*__inference_reshape_14_layer_call_fn_61381

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_14_layer_call_and_return_conditional_losses_592372
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ë
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_62139

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
§
Ø
'__inference_model_8_layer_call_fn_59880
input_12
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_12

E
)__inference_dropout_5_layer_call_fn_62149

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_597062
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ïb
ø
(T-GCN-WX_model_8_lstm_3_while_body_58508L
Ht_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_loop_counterR
Nt_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_maximum_iterations-
)t_gcn_wx_model_8_lstm_3_while_placeholder/
+t_gcn_wx_model_8_lstm_3_while_placeholder_1/
+t_gcn_wx_model_8_lstm_3_while_placeholder_2/
+t_gcn_wx_model_8_lstm_3_while_placeholder_3K
Gt_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_strided_slice_1_0
t_gcn_wx_model_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0N
Jt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0P
Lt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0O
Kt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
&t_gcn_wx_model_8_lstm_3_while_identity,
(t_gcn_wx_model_8_lstm_3_while_identity_1,
(t_gcn_wx_model_8_lstm_3_while_identity_2,
(t_gcn_wx_model_8_lstm_3_while_identity_3,
(t_gcn_wx_model_8_lstm_3_while_identity_4,
(t_gcn_wx_model_8_lstm_3_while_identity_5I
Et_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_strided_slice_1
t_gcn_wx_model_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensorL
Ht_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceN
Jt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceM
It_gcn_wx_model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource¢@T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp¢?T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp¢AT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpó
OT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2Q
OT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeä
AT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemt_gcn_wx_model_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0)t_gcn_wx_model_8_lstm_3_while_placeholderXT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02C
AT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem
?T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOpJt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype02A
?T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp´
0T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMulMatMulHT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0GT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul
AT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOpLt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02C
AT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp
2T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1MatMul+t_gcn_wx_model_8_lstm_3_while_placeholder_2IT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1
-T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/addAddV2:T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul:product:0<T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/add
@T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOpKt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02B
@T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp¡
1T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAddBiasAdd1T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/add:z:0HT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd¤
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :21
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Const¸
9T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2;
9T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/split/split_dimë
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/splitSplitBT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/split/split_dim:output:0:T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split21
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/splitÞ
1T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/SigmoidSigmoid8T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ23
1T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoidâ
3T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid8T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ25
3T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoid_1þ
-T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mulMul7T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoid_1:y:0+t_gcn_wx_model_8_lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2/
-T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul_1Mul5T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoid:y:08T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ21
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul_1
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/add_1AddV21T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul:z:03T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ21
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/add_1â
3T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid8T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ25
3T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoid_2
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul_2Mul7T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/Sigmoid_2:y:03T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ21
/T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul_2×
BT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+t_gcn_wx_model_8_lstm_3_while_placeholder_1)t_gcn_wx_model_8_lstm_3_while_placeholder3T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02D
BT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Write/TensorListSetItem
#T-GCN-WX/model_8/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#T-GCN-WX/model_8/lstm_3/while/add/yÉ
!T-GCN-WX/model_8/lstm_3/while/addAddV2)t_gcn_wx_model_8_lstm_3_while_placeholder,T-GCN-WX/model_8/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2#
!T-GCN-WX/model_8/lstm_3/while/add
%T-GCN-WX/model_8/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%T-GCN-WX/model_8/lstm_3/while/add_1/yî
#T-GCN-WX/model_8/lstm_3/while/add_1AddV2Ht_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_loop_counter.T-GCN-WX/model_8/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2%
#T-GCN-WX/model_8/lstm_3/while/add_1ï
&T-GCN-WX/model_8/lstm_3/while/IdentityIdentity'T-GCN-WX/model_8/lstm_3/while/add_1:z:0A^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp@^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpB^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&T-GCN-WX/model_8/lstm_3/while/Identity
(T-GCN-WX/model_8/lstm_3/while/Identity_1IdentityNt_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_maximum_iterationsA^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp@^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpB^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(T-GCN-WX/model_8/lstm_3/while/Identity_1ñ
(T-GCN-WX/model_8/lstm_3/while/Identity_2Identity%T-GCN-WX/model_8/lstm_3/while/add:z:0A^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp@^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpB^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(T-GCN-WX/model_8/lstm_3/while/Identity_2
(T-GCN-WX/model_8/lstm_3/while/Identity_3IdentityRT-GCN-WX/model_8/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0A^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp@^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpB^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(T-GCN-WX/model_8/lstm_3/while/Identity_3
(T-GCN-WX/model_8/lstm_3/while/Identity_4Identity3T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/mul_2:z:0A^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp@^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpB^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(T-GCN-WX/model_8/lstm_3/while/Identity_4
(T-GCN-WX/model_8/lstm_3/while/Identity_5Identity3T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/add_1:z:0A^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp@^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpB^T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(T-GCN-WX/model_8/lstm_3/while/Identity_5"Y
&t_gcn_wx_model_8_lstm_3_while_identity/T-GCN-WX/model_8/lstm_3/while/Identity:output:0"]
(t_gcn_wx_model_8_lstm_3_while_identity_11T-GCN-WX/model_8/lstm_3/while/Identity_1:output:0"]
(t_gcn_wx_model_8_lstm_3_while_identity_21T-GCN-WX/model_8/lstm_3/while/Identity_2:output:0"]
(t_gcn_wx_model_8_lstm_3_while_identity_31T-GCN-WX/model_8/lstm_3/while/Identity_3:output:0"]
(t_gcn_wx_model_8_lstm_3_while_identity_41T-GCN-WX/model_8/lstm_3/while/Identity_4:output:0"]
(t_gcn_wx_model_8_lstm_3_while_identity_51T-GCN-WX/model_8/lstm_3/while/Identity_5:output:0"
It_gcn_wx_model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resourceKt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"
Jt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceLt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"
Ht_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceJt_gcn_wx_model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"
Et_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_strided_slice_1Gt_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_strided_slice_1_0"
t_gcn_wx_model_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensort_gcn_wx_model_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2
@T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp@T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2
?T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp?T-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2
AT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpAT-GCN-WX/model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ø
|
'__inference_dense_8_layer_call_fn_60820

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_599142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¾&
Í
B__inference_model_8_layer_call_and_return_conditional_losses_59777
input_12-
)fixed_adjacency_graph_convolution_3_59753-
)fixed_adjacency_graph_convolution_3_59755-
)fixed_adjacency_graph_convolution_3_59757
lstm_3_59763
lstm_3_59765
lstm_3_59767
dense_7_59771
dense_7_59773
identity¢dense_7/StatefulPartitionedCall¢;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¶
tf.expand_dims_3/ExpandDims
ExpandDimsinput_12(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_3/ExpandDimsû
reshape_14/PartitionedCallPartitionedCall$tf.expand_dims_3/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_14_layer_call_and_return_conditional_losses_592372
reshape_14/PartitionedCallæ
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0)fixed_adjacency_graph_convolution_3_59753)fixed_adjacency_graph_convolution_3_59755)fixed_adjacency_graph_convolution_3_59757*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *g
fbR`
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_592982=
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall
reshape_15/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_15_layer_call_and_return_conditional_losses_593322
reshape_15/PartitionedCallû
permute_3/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_permute_3_layer_call_and_return_conditional_losses_586062
permute_3/PartitionedCallù
reshape_16/PartitionedCallPartitionedCall"permute_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_593542
reshape_16/PartitionedCallµ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0lstm_3_59763lstm_3_59765lstm_3_59767*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_596592 
lstm_3/StatefulPartitionedCallø
dropout_5/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_597062
dropout_5/PartitionedCall¨
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_59771dense_7_59773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_597302!
dense_7/StatefulPartitionedCallý
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_3/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_12
Ú
|
'__inference_dense_7_layer_call_fn_62169

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_597302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Á
Ë
+__inference_lstm_cell_3_layer_call_fn_62265

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_587142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1

Û
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_62231

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2©
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity­

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1­

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1
¬
F
*__inference_reshape_16_layer_call_fn_61482

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_593542
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
F:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
Ç
©
C__inference_fixed_adjacency_graph_convolution_3_layer_call_fn_61445
features
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallfeaturesunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *g
fbR`
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_592982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
features
­
ä
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60068

inputs
dense_8_60044
dense_8_60046
model_8_60050
model_8_60052
model_8_60054
model_8_60056
model_8_60058
model_8_60060
model_8_60062
model_8_60064
identity¢dense_8/StatefulPartitionedCall¢model_8/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_60044dense_8_60046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_599142!
dense_8/StatefulPartitionedCallÿ
reshape_13/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_599432
reshape_13/PartitionedCall
model_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0model_8_60050model_8_60052model_8_60054model_8_60056model_8_60058model_8_60060model_8_60062model_8_60064*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598102!
model_8/StatefulPartitionedCallÀ
IdentityIdentity(model_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ë
õ
(__inference_T-GCN-WX_layer_call_fn_60781

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_601202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ñ	
Û
B__inference_dense_7_layer_call_and_return_conditional_losses_59730

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ÈF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
 ó
È
B__inference_model_8_layer_call_and_return_conditional_losses_61083

inputsG
Cfixed_adjacency_graph_convolution_3_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_3_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_3_add_readvariableop_resource5
1lstm_3_lstm_cell_3_matmul_readvariableop_resource7
3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource6
2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢6fixed_adjacency_graph_convolution_3/add/ReadVariableOp¢>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp¢>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp¢)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp¢(lstm_3/lstm_cell_3/MatMul/ReadVariableOp¢*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp¢lstm_3/while
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim´
tf.expand_dims_3/ExpandDims
ExpandDimsinputs(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_3/ExpandDimsx
reshape_14/ShapeShape$tf.expand_dims_3/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_14/Shape
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stack
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2¤
reshape_14/strided_sliceStridedSlicereshape_14/Shape:output:0'reshape_14/strided_slice/stack:output:0)reshape_14/strided_slice/stack_1:output:0)reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_14/strided_slicez
reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_14/Reshape/shape/1z
reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/2×
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shape²
reshape_14/ReshapeReshape$tf.expand_dims_3/ExpandDims:output:0!reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_14/Reshape½
2fixed_adjacency_graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_3/transpose/permû
-fixed_adjacency_graph_convolution_3/transpose	Transposereshape_14/Reshape:output:0;fixed_adjacency_graph_convolution_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2/
-fixed_adjacency_graph_convolution_3/transpose·
)fixed_adjacency_graph_convolution_3/ShapeShape1fixed_adjacency_graph_convolution_3/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_3/ShapeÈ
+fixed_adjacency_graph_convolution_3/unstackUnpack2fixed_adjacency_graph_convolution_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_3/unstackü
:fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOp«
+fixed_adjacency_graph_convolution_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_3/Shape_1Ì
-fixed_adjacency_graph_convolution_3/unstack_1Unpack4fixed_adjacency_graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_3/unstack_1·
1fixed_adjacency_graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   23
1fixed_adjacency_graph_convolution_3/Reshape/shape
+fixed_adjacency_graph_convolution_3/ReshapeReshape1fixed_adjacency_graph_convolution_3/transpose:y:0:fixed_adjacency_graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2-
+fixed_adjacency_graph_convolution_3/Reshape
>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp½
4fixed_adjacency_graph_convolution_3/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_3/transpose_1/perm
/fixed_adjacency_graph_convolution_3/transpose_1	TransposeFfixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_3/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_3/transpose_1»
3fixed_adjacency_graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ25
3fixed_adjacency_graph_convolution_3/Reshape_1/shape
-fixed_adjacency_graph_convolution_3/Reshape_1Reshape3fixed_adjacency_graph_convolution_3/transpose_1:y:0<fixed_adjacency_graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_3/Reshape_1
*fixed_adjacency_graph_convolution_3/MatMulMatMul4fixed_adjacency_graph_convolution_3/Reshape:output:06fixed_adjacency_graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2,
*fixed_adjacency_graph_convolution_3/MatMul°
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/1°
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/2Ö
3fixed_adjacency_graph_convolution_3/Reshape_2/shapePack4fixed_adjacency_graph_convolution_3/unstack:output:0>fixed_adjacency_graph_convolution_3/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_3/Reshape_2/shape
-fixed_adjacency_graph_convolution_3/Reshape_2Reshape4fixed_adjacency_graph_convolution_3/MatMul:product:0<fixed_adjacency_graph_convolution_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2/
-fixed_adjacency_graph_convolution_3/Reshape_2Á
4fixed_adjacency_graph_convolution_3/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_3/transpose_2/perm
/fixed_adjacency_graph_convolution_3/transpose_2	Transpose6fixed_adjacency_graph_convolution_3/Reshape_2:output:0=fixed_adjacency_graph_convolution_3/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF21
/fixed_adjacency_graph_convolution_3/transpose_2½
+fixed_adjacency_graph_convolution_3/Shape_2Shape3fixed_adjacency_graph_convolution_3/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_3/Shape_2Î
-fixed_adjacency_graph_convolution_3/unstack_2Unpack4fixed_adjacency_graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_3/unstack_2ü
:fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02<
:fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOp«
+fixed_adjacency_graph_convolution_3/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2-
+fixed_adjacency_graph_convolution_3/Shape_3Ì
-fixed_adjacency_graph_convolution_3/unstack_3Unpack4fixed_adjacency_graph_convolution_3/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_3/unstack_3»
3fixed_adjacency_graph_convolution_3/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   25
3fixed_adjacency_graph_convolution_3/Reshape_3/shape
-fixed_adjacency_graph_convolution_3/Reshape_3Reshape3fixed_adjacency_graph_convolution_3/transpose_2:y:0<fixed_adjacency_graph_convolution_3/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-fixed_adjacency_graph_convolution_3/Reshape_3
>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02@
>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp½
4fixed_adjacency_graph_convolution_3/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_3/transpose_3/perm
/fixed_adjacency_graph_convolution_3/transpose_3	TransposeFfixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_3/transpose_3/perm:output:0*
T0*
_output_shapes

:
21
/fixed_adjacency_graph_convolution_3/transpose_3»
3fixed_adjacency_graph_convolution_3/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ25
3fixed_adjacency_graph_convolution_3/Reshape_4/shape
-fixed_adjacency_graph_convolution_3/Reshape_4Reshape3fixed_adjacency_graph_convolution_3/transpose_3:y:0<fixed_adjacency_graph_convolution_3/Reshape_4/shape:output:0*
T0*
_output_shapes

:
2/
-fixed_adjacency_graph_convolution_3/Reshape_4
,fixed_adjacency_graph_convolution_3/MatMul_1MatMul6fixed_adjacency_graph_convolution_3/Reshape_3:output:06fixed_adjacency_graph_convolution_3/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2.
,fixed_adjacency_graph_convolution_3/MatMul_1°
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/1°
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
27
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/2Ø
3fixed_adjacency_graph_convolution_3/Reshape_5/shapePack6fixed_adjacency_graph_convolution_3/unstack_2:output:0>fixed_adjacency_graph_convolution_3/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_3/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_3/Reshape_5/shape
-fixed_adjacency_graph_convolution_3/Reshape_5Reshape6fixed_adjacency_graph_convolution_3/MatMul_1:product:0<fixed_adjacency_graph_convolution_3/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2/
-fixed_adjacency_graph_convolution_3/Reshape_5ð
6fixed_adjacency_graph_convolution_3/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_3_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_3/add/ReadVariableOp
'fixed_adjacency_graph_convolution_3/addAddV26fixed_adjacency_graph_convolution_3/Reshape_5:output:0>fixed_adjacency_graph_convolution_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2)
'fixed_adjacency_graph_convolution_3/add
reshape_15/ShapeShape+fixed_adjacency_graph_convolution_3/add:z:0*
T0*
_output_shapes
:2
reshape_15/Shape
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2¤
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_15/Reshape/shape/1
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_15/Reshape/shape/2z
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/3ü
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape½
reshape_15/ReshapeReshape+fixed_adjacency_graph_convolution_3/add:z:0!reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
reshape_15/Reshape
permute_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_3/transpose/perm±
permute_3/transpose	Transposereshape_15/Reshape:output:0!permute_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
permute_3/transposek
reshape_16/ShapeShapepermute_3/transpose:y:0*
T0*
_output_shapes
:2
reshape_16/Shape
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2¤
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slice
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_16/Reshape/shape/2×
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape¥
reshape_16/ReshapeReshapepermute_3/transpose:y:0!reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
reshape_16/Reshapeg
lstm_3/ShapeShapereshape_16/Reshape:output:0*
T0*
_output_shapes
:2
lstm_3/Shape
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicek
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros/mul/y
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_3/zeros/Less/y
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessq
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros/packed/1
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/zeroso
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros_1/mul/y
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_3/zeros_1/Less/y
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lessu
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros_1/packed/1¥
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/zeros_1
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm¤
lstm_3/transpose	Transposereshape_16/Reshape:output:0lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stack
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_3/TensorArrayV2/element_shapeÎ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2Í
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensor
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stack
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2¦
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
lstm_3/strided_slice_2Ç
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02*
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpÆ
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/MatMulÎ
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02,
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpÂ
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/MatMul_1¸
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/MatMul:product:0%lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/addÆ
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02+
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpÅ
lstm_3/lstm_cell_3/BiasAddBiasAddlstm_3/lstm_cell_3/add:z:01lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/BiasAddv
lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/lstm_cell_3/Const
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dim
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0#lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_3/lstm_cell_3/split
lstm_3/lstm_cell_3/SigmoidSigmoid!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/Sigmoid
lstm_3/lstm_cell_3/Sigmoid_1Sigmoid!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/Sigmoid_1¥
lstm_3/lstm_cell_3/mulMul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/mul±
lstm_3/lstm_cell_3/mul_1Mullstm_3/lstm_cell_3/Sigmoid:y:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/mul_1ª
lstm_3/lstm_cell_3/add_1AddV2lstm_3/lstm_cell_3/mul:z:0lstm_3/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/add_1
lstm_3/lstm_cell_3/Sigmoid_2Sigmoid!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/Sigmoid_2®
lstm_3/lstm_cell_3/mul_2Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0lstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/mul_2
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2&
$lstm_3/TensorArrayV2_1/element_shapeÔ
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/time
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counterÖ
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_3_matmul_readvariableop_resource3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_3_while_body_60985*#
condR
lstm_3_while_cond_60984*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
lstm_3/whileÃ
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStack
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_3/strided_slice_3/stack
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2Å
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
lstm_3/strided_slice_3
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/permÂ
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimew
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_5/dropout/Const«
dropout_5/dropout/MulMullstm_3/strided_slice_3:output:0 dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_5/dropout/Mul
dropout_5/dropout/ShapeShapelstm_3/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/ShapeÓ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_5/dropout/GreaterEqual/yç
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_5/dropout/Cast£
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_5/dropout/Mul_1¦
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	ÈF*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_7/BiasAddy
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_7/Sigmoidö
IdentityIdentitydense_7/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_3/add/ReadVariableOp?^fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp*^lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_3/MatMul/ReadVariableOp+^lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^lstm_3/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_3/add/ReadVariableOp6fixed_adjacency_graph_convolution_3/add/ReadVariableOp2
>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp2
>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp2V
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp(lstm_3/lstm_cell_3/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Y
ì
A__inference_lstm_3_layer_call_and_return_conditional_losses_62100
inputs_0.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity¢"lstm_cell_3/BiasAdd/ReadVariableOp¢!lstm_cell_3/MatMul/ReadVariableOp¢#lstm_cell_3/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02#
!lstm_cell_3/MatMul/ReadVariableOpª
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul¹
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¦
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul_1
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimó
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_cell_3/split
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_1
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_1
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/add_1
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_2
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterí
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62017*
condR
while_cond_62016*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeä
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0
ë
á
B__inference_dense_8_layer_call_and_return_conditional_losses_60811

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
BiasAdd 
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
î'
ñ
B__inference_model_8_layer_call_and_return_conditional_losses_59747
input_12-
)fixed_adjacency_graph_convolution_3_59311-
)fixed_adjacency_graph_convolution_3_59313-
)fixed_adjacency_graph_convolution_3_59315
lstm_3_59682
lstm_3_59684
lstm_3_59686
dense_7_59741
dense_7_59743
identity¢dense_7/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¶
tf.expand_dims_3/ExpandDims
ExpandDimsinput_12(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_3/ExpandDimsû
reshape_14/PartitionedCallPartitionedCall$tf.expand_dims_3/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_14_layer_call_and_return_conditional_losses_592372
reshape_14/PartitionedCallæ
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0)fixed_adjacency_graph_convolution_3_59311)fixed_adjacency_graph_convolution_3_59313)fixed_adjacency_graph_convolution_3_59315*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *g
fbR`
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_592982=
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall
reshape_15/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_15_layer_call_and_return_conditional_losses_593322
reshape_15/PartitionedCallû
permute_3/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_permute_3_layer_call_and_return_conditional_losses_586062
permute_3/PartitionedCallù
reshape_16/PartitionedCallPartitionedCall"permute_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_593542
reshape_16/PartitionedCallµ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0lstm_3_59682lstm_3_59684lstm_3_59686*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_595102 
lstm_3/StatefulPartitionedCall
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_597012#
!dropout_5/StatefulPartitionedCall°
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_7_59741dense_7_59743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_597302!
dense_7/StatefulPartitionedCall¡
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_3/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_12
É
ò
#__inference_signature_wrapper_60178
input_11
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_585992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_11
@
ô
while_body_61548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource¢(while/lstm_cell_3/BiasAdd/ReadVariableOp¢'while/lstm_cell_3/MatMul/ReadVariableOp¢)while/lstm_cell_3/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOpÔ
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMulÍ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp½
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMul_1´
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/addÅ
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpÁ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
while/lstm_cell_3/split
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_1
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul­
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_1¦
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/add_1
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_2ª
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ÒX
ê
A__inference_lstm_3_layer_call_and_return_conditional_losses_61780

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity¢"lstm_cell_3/BiasAdd/ReadVariableOp¢!lstm_cell_3/MatMul/ReadVariableOp¢#lstm_cell_3/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02#
!lstm_cell_3/MatMul/ReadVariableOpª
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul¹
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¦
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul_1
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimó
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_cell_3/split
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_1
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_1
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/add_1
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_2
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterí
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_61697*
condR
while_cond_61696*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeä
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
F:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
J
Ô	
lstm_3_while_body_60985*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0?
;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0>
:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor;
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource=
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource<
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource¢/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp¢.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp¢0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpÑ
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype020
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpð
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lstm_3/while/lstm_cell_3/MatMulâ
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype022
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpÙ
!lstm_3/while/lstm_cell_3/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lstm_3/while/lstm_cell_3/MatMul_1Ð
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/MatMul:product:0+lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/while/lstm_cell_3/addÚ
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype021
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpÝ
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd lstm_3/while/lstm_cell_3/add:z:07lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 lstm_3/while/lstm_cell_3/BiasAdd
lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_3/while/lstm_cell_3/Const
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim§
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:0)lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2 
lstm_3/while/lstm_cell_3/split«
 lstm_3/while/lstm_cell_3/SigmoidSigmoid'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 lstm_3/while/lstm_cell_3/Sigmoid¯
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm_3/while/lstm_cell_3/Sigmoid_1º
lstm_3/while/lstm_cell_3/mulMul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/while/lstm_cell_3/mulÉ
lstm_3/while/lstm_cell_3/mul_1Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_3/while/lstm_cell_3/mul_1Â
lstm_3/while/lstm_cell_3/add_1AddV2 lstm_3/while/lstm_cell_3/mul:z:0"lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_3/while/lstm_cell_3/add_1¯
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm_3/while/lstm_cell_3/Sigmoid_2Æ
lstm_3/while/lstm_cell_3/mul_2Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_3/while/lstm_cell_3/mul_2
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/y
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/y
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity£
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations0^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1
lstm_3/while/Identity_2Identitylstm_3/while/add:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2¸
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3«
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_2:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/while/Identity_4«
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_1:z:00^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/while/Identity_5"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"Ä
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2b
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ÎD
Ó
A__inference_lstm_3_layer_call_and_return_conditional_losses_59209

inputs
lstm_cell_3_59127
lstm_cell_3_59129
lstm_cell_3_59131
identity¢#lstm_cell_3/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_59127lstm_cell_3_59129lstm_cell_3_59131*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_587142%
#lstm_cell_3/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_59127lstm_cell_3_59129lstm_cell_3_59131*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_59140*
condR
while_cond_59139*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_3/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
@
ô
while_body_61697
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource¢(while/lstm_cell_3/BiasAdd/ReadVariableOp¢'while/lstm_cell_3/MatMul/ReadVariableOp¢)while/lstm_cell_3/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOpÔ
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMulÍ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp½
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMul_1´
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/addÅ
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpÁ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
while/lstm_cell_3/split
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_1
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul­
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_1¦
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/add_1
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_2ª
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
î
ê
model_8_lstm_3_while_cond_60639:
6model_8_lstm_3_while_model_8_lstm_3_while_loop_counter@
<model_8_lstm_3_while_model_8_lstm_3_while_maximum_iterations$
 model_8_lstm_3_while_placeholder&
"model_8_lstm_3_while_placeholder_1&
"model_8_lstm_3_while_placeholder_2&
"model_8_lstm_3_while_placeholder_3<
8model_8_lstm_3_while_less_model_8_lstm_3_strided_slice_1Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60639___redundant_placeholder0Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60639___redundant_placeholder1Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60639___redundant_placeholder2Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60639___redundant_placeholder3!
model_8_lstm_3_while_identity
»
model_8/lstm_3/while/LessLess model_8_lstm_3_while_placeholder8model_8_lstm_3_while_less_model_8_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
model_8/lstm_3/while/Less
model_8/lstm_3/while/IdentityIdentitymodel_8/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
model_8/lstm_3/while/Identity"G
model_8_lstm_3_while_identity&model_8/lstm_3/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
î
a
E__inference_reshape_13_layer_call_and_return_conditional_losses_59943

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¾,
¹
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_59298
features#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource
add_readvariableop_resource
identity¢add/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposefeaturestranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:FF2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:
2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:
2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
	Reshape_5
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
add®
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2(
add/ReadVariableOpadd/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
features
@
ô
while_body_59427
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource¢(while/lstm_cell_3/BiasAdd/ReadVariableOp¢'while/lstm_cell_3/MatMul/ReadVariableOp¢)while/lstm_cell_3/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOpÔ
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMulÍ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp½
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMul_1´
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/addÅ
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpÁ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
while/lstm_cell_3/split
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_1
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul­
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_1¦
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/add_1
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_2ª
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ë
õ
(__inference_T-GCN-WX_layer_call_fn_60756

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_600682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ÒX
ê
A__inference_lstm_3_layer_call_and_return_conditional_losses_59659

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity¢"lstm_cell_3/BiasAdd/ReadVariableOp¢!lstm_cell_3/MatMul/ReadVariableOp¢#lstm_cell_3/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02#
!lstm_cell_3/MatMul/ReadVariableOpª
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul¹
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¦
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul_1
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimó
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_cell_3/split
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_1
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_1
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/add_1
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_2
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterí
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_59576*
condR
while_cond_59575*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeä
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
F:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs

E
)__inference_permute_3_layer_call_fn_58612

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_permute_3_layer_call_and_return_conditional_losses_586062
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
¾
while_cond_59426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_59426___redundant_placeholder03
/while_while_cond_59426___redundant_placeholder13
/while_while_cond_59426___redundant_placeholder23
/while_while_cond_59426___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
è'
ï
B__inference_model_8_layer_call_and_return_conditional_losses_59810

inputs-
)fixed_adjacency_graph_convolution_3_59786-
)fixed_adjacency_graph_convolution_3_59788-
)fixed_adjacency_graph_convolution_3_59790
lstm_3_59796
lstm_3_59798
lstm_3_59800
dense_7_59804
dense_7_59806
identity¢dense_7/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim´
tf.expand_dims_3/ExpandDims
ExpandDimsinputs(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_3/ExpandDimsû
reshape_14/PartitionedCallPartitionedCall$tf.expand_dims_3/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_14_layer_call_and_return_conditional_losses_592372
reshape_14/PartitionedCallæ
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0)fixed_adjacency_graph_convolution_3_59786)fixed_adjacency_graph_convolution_3_59788)fixed_adjacency_graph_convolution_3_59790*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *g
fbR`
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_592982=
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall
reshape_15/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_15_layer_call_and_return_conditional_losses_593322
reshape_15/PartitionedCallû
permute_3/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_permute_3_layer_call_and_return_conditional_losses_586062
permute_3/PartitionedCallù
reshape_16/PartitionedCallPartitionedCall"permute_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_593542
reshape_16/PartitionedCallµ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0lstm_3_59796lstm_3_59798lstm_3_59800*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_595102 
lstm_3/StatefulPartitionedCall
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_597012#
!dropout_5/StatefulPartitionedCall°
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_7_59804dense_7_59806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_597302!
dense_7/StatefulPartitionedCall¡
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_3/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
÷
a
E__inference_reshape_15_layer_call_and_return_conditional_losses_59332

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1m
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿF
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF

 
_user_specified_nameinputs
¬
F
*__inference_reshape_13_layer_call_fn_60838

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_599432
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Y
ì
A__inference_lstm_3_layer_call_and_return_conditional_losses_61951
inputs_0.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity¢"lstm_cell_3/BiasAdd/ReadVariableOp¢!lstm_cell_3/MatMul/ReadVariableOp¢#lstm_cell_3/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02#
!lstm_cell_3/MatMul/ReadVariableOpª
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul¹
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¦
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul_1
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimó
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_cell_3/split
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_1
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_1
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/add_1
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_2
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterí
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_61868*
condR
while_cond_61867*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeä
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0
áU
Ô
model_8_lstm_3_while_body_60640:
6model_8_lstm_3_while_model_8_lstm_3_while_loop_counter@
<model_8_lstm_3_while_model_8_lstm_3_while_maximum_iterations$
 model_8_lstm_3_while_placeholder&
"model_8_lstm_3_while_placeholder_1&
"model_8_lstm_3_while_placeholder_2&
"model_8_lstm_3_while_placeholder_39
5model_8_lstm_3_while_model_8_lstm_3_strided_slice_1_0u
qmodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0E
Amodel_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0G
Cmodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0F
Bmodel_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0!
model_8_lstm_3_while_identity#
model_8_lstm_3_while_identity_1#
model_8_lstm_3_while_identity_2#
model_8_lstm_3_while_identity_3#
model_8_lstm_3_while_identity_4#
model_8_lstm_3_while_identity_57
3model_8_lstm_3_while_model_8_lstm_3_strided_slice_1s
omodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensorC
?model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceE
Amodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceD
@model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource¢7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp¢6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp¢8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpá
Fmodel_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2H
Fmodel_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0 model_8_lstm_3_while_placeholderOmodel_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02:
8model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItemó
6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOpAmodel_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype028
6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp
'model_8/lstm_3/while/lstm_cell_3/MatMulMatMul?model_8/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0>model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'model_8/lstm_3/while/lstm_cell_3/MatMulú
8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOpCmodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02:
8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpù
)model_8/lstm_3/while/lstm_cell_3/MatMul_1MatMul"model_8_lstm_3_while_placeholder_2@model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)model_8/lstm_3/while/lstm_cell_3/MatMul_1ð
$model_8/lstm_3/while/lstm_cell_3/addAddV21model_8/lstm_3/while/lstm_cell_3/MatMul:product:03model_8/lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$model_8/lstm_3/while/lstm_cell_3/addò
7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOpBmodel_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype029
7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpý
(model_8/lstm_3/while/lstm_cell_3/BiasAddBiasAdd(model_8/lstm_3/while/lstm_cell_3/add:z:0?model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(model_8/lstm_3/while/lstm_cell_3/BiasAdd
&model_8/lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_8/lstm_3/while/lstm_cell_3/Const¦
0model_8/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_8/lstm_3/while/lstm_cell_3/split/split_dimÇ
&model_8/lstm_3/while/lstm_cell_3/splitSplit9model_8/lstm_3/while/lstm_cell_3/split/split_dim:output:01model_8/lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2(
&model_8/lstm_3/while/lstm_cell_3/splitÃ
(model_8/lstm_3/while/lstm_cell_3/SigmoidSigmoid/model_8/lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(model_8/lstm_3/while/lstm_cell_3/SigmoidÇ
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid/model_8/lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2,
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_1Ú
$model_8/lstm_3/while/lstm_cell_3/mulMul.model_8/lstm_3/while/lstm_cell_3/Sigmoid_1:y:0"model_8_lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$model_8/lstm_3/while/lstm_cell_3/mulé
&model_8/lstm_3/while/lstm_cell_3/mul_1Mul,model_8/lstm_3/while/lstm_cell_3/Sigmoid:y:0/model_8/lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&model_8/lstm_3/while/lstm_cell_3/mul_1â
&model_8/lstm_3/while/lstm_cell_3/add_1AddV2(model_8/lstm_3/while/lstm_cell_3/mul:z:0*model_8/lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&model_8/lstm_3/while/lstm_cell_3/add_1Ç
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid/model_8/lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2,
*model_8/lstm_3/while/lstm_cell_3/Sigmoid_2æ
&model_8/lstm_3/while/lstm_cell_3/mul_2Mul.model_8/lstm_3/while/lstm_cell_3/Sigmoid_2:y:0*model_8/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&model_8/lstm_3/while/lstm_cell_3/mul_2ª
9model_8/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_8_lstm_3_while_placeholder_1 model_8_lstm_3_while_placeholder*model_8/lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9model_8/lstm_3/while/TensorArrayV2Write/TensorListSetItemz
model_8/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_8/lstm_3/while/add/y¥
model_8/lstm_3/while/addAddV2 model_8_lstm_3_while_placeholder#model_8/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/while/add~
model_8/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_8/lstm_3/while/add_1/yÁ
model_8/lstm_3/while/add_1AddV26model_8_lstm_3_while_model_8_lstm_3_while_loop_counter%model_8/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/while/add_1¹
model_8/lstm_3/while/IdentityIdentitymodel_8/lstm_3/while/add_1:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
model_8/lstm_3/while/IdentityÛ
model_8/lstm_3/while/Identity_1Identity<model_8_lstm_3_while_model_8_lstm_3_while_maximum_iterations8^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_3/while/Identity_1»
model_8/lstm_3/while/Identity_2Identitymodel_8/lstm_3/while/add:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_3/while/Identity_2è
model_8/lstm_3/while/Identity_3IdentityImodel_8/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_3/while/Identity_3Û
model_8/lstm_3/while/Identity_4Identity*model_8/lstm_3/while/lstm_cell_3/mul_2:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
model_8/lstm_3/while/Identity_4Û
model_8/lstm_3/while/Identity_5Identity*model_8/lstm_3/while/lstm_cell_3/add_1:z:08^model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7^model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp9^model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
model_8/lstm_3/while/Identity_5"G
model_8_lstm_3_while_identity&model_8/lstm_3/while/Identity:output:0"K
model_8_lstm_3_while_identity_1(model_8/lstm_3/while/Identity_1:output:0"K
model_8_lstm_3_while_identity_2(model_8/lstm_3/while/Identity_2:output:0"K
model_8_lstm_3_while_identity_3(model_8/lstm_3/while/Identity_3:output:0"K
model_8_lstm_3_while_identity_4(model_8/lstm_3/while/Identity_4:output:0"K
model_8_lstm_3_while_identity_5(model_8/lstm_3/while/Identity_5:output:0"
@model_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resourceBmodel_8_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"
Amodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceCmodel_8_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"
?model_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceAmodel_8_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"l
3model_8_lstm_3_while_model_8_lstm_3_strided_slice_15model_8_lstm_3_while_model_8_lstm_3_strided_slice_1_0"ä
omodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensorqmodel_8_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2r
7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp7model_8/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2p
6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp6model_8/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2t
8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp8model_8/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ñ	
Û
B__inference_dense_7_layer_call_and_return_conditional_losses_62160

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ÈF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ÎD
Ó
A__inference_lstm_3_layer_call_and_return_conditional_losses_59077

inputs
lstm_cell_3_58995
lstm_cell_3_58997
lstm_cell_3_58999
identity¢#lstm_cell_3/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_58995lstm_cell_3_58997lstm_cell_3_58999*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_586832%
#lstm_cell_3/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_58995lstm_cell_3_58997lstm_cell_3_58999*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_59008*
condR
while_cond_59007*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_3/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
î
a
E__inference_reshape_14_layer_call_and_return_conditional_losses_59237

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ã

 __inference__wrapped_model_58599
input_116
2t_gcn_wx_dense_8_tensordot_readvariableop_resource4
0t_gcn_wx_dense_8_biasadd_readvariableop_resourceX
Tt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resourceX
Tt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resourceT
Pt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_add_readvariableop_resourceF
Bt_gcn_wx_model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resourceH
Dt_gcn_wx_model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resourceG
Ct_gcn_wx_model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource;
7t_gcn_wx_model_8_dense_7_matmul_readvariableop_resource<
8t_gcn_wx_model_8_dense_7_biasadd_readvariableop_resource
identity¢'T-GCN-WX/dense_8/BiasAdd/ReadVariableOp¢)T-GCN-WX/dense_8/Tensordot/ReadVariableOp¢/T-GCN-WX/model_8/dense_7/BiasAdd/ReadVariableOp¢.T-GCN-WX/model_8/dense_7/MatMul/ReadVariableOp¢GT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp¢OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp¢OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp¢:T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp¢9T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp¢;T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp¢T-GCN-WX/model_8/lstm_3/whileÉ
)T-GCN-WX/dense_8/Tensordot/ReadVariableOpReadVariableOp2t_gcn_wx_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02+
)T-GCN-WX/dense_8/Tensordot/ReadVariableOp
T-GCN-WX/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
T-GCN-WX/dense_8/Tensordot/axes
T-GCN-WX/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
T-GCN-WX/dense_8/Tensordot/free|
 T-GCN-WX/dense_8/Tensordot/ShapeShapeinput_11*
T0*
_output_shapes
:2"
 T-GCN-WX/dense_8/Tensordot/Shape
(T-GCN-WX/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(T-GCN-WX/dense_8/Tensordot/GatherV2/axis¦
#T-GCN-WX/dense_8/Tensordot/GatherV2GatherV2)T-GCN-WX/dense_8/Tensordot/Shape:output:0(T-GCN-WX/dense_8/Tensordot/free:output:01T-GCN-WX/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#T-GCN-WX/dense_8/Tensordot/GatherV2
*T-GCN-WX/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*T-GCN-WX/dense_8/Tensordot/GatherV2_1/axis¬
%T-GCN-WX/dense_8/Tensordot/GatherV2_1GatherV2)T-GCN-WX/dense_8/Tensordot/Shape:output:0(T-GCN-WX/dense_8/Tensordot/axes:output:03T-GCN-WX/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%T-GCN-WX/dense_8/Tensordot/GatherV2_1
 T-GCN-WX/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 T-GCN-WX/dense_8/Tensordot/ConstÄ
T-GCN-WX/dense_8/Tensordot/ProdProd,T-GCN-WX/dense_8/Tensordot/GatherV2:output:0)T-GCN-WX/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
T-GCN-WX/dense_8/Tensordot/Prod
"T-GCN-WX/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"T-GCN-WX/dense_8/Tensordot/Const_1Ì
!T-GCN-WX/dense_8/Tensordot/Prod_1Prod.T-GCN-WX/dense_8/Tensordot/GatherV2_1:output:0+T-GCN-WX/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!T-GCN-WX/dense_8/Tensordot/Prod_1
&T-GCN-WX/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&T-GCN-WX/dense_8/Tensordot/concat/axis
!T-GCN-WX/dense_8/Tensordot/concatConcatV2(T-GCN-WX/dense_8/Tensordot/free:output:0(T-GCN-WX/dense_8/Tensordot/axes:output:0/T-GCN-WX/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!T-GCN-WX/dense_8/Tensordot/concatÐ
 T-GCN-WX/dense_8/Tensordot/stackPack(T-GCN-WX/dense_8/Tensordot/Prod:output:0*T-GCN-WX/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 T-GCN-WX/dense_8/Tensordot/stackÉ
$T-GCN-WX/dense_8/Tensordot/transpose	Transposeinput_11*T-GCN-WX/dense_8/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2&
$T-GCN-WX/dense_8/Tensordot/transposeã
"T-GCN-WX/dense_8/Tensordot/ReshapeReshape(T-GCN-WX/dense_8/Tensordot/transpose:y:0)T-GCN-WX/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"T-GCN-WX/dense_8/Tensordot/Reshapeâ
!T-GCN-WX/dense_8/Tensordot/MatMulMatMul+T-GCN-WX/dense_8/Tensordot/Reshape:output:01T-GCN-WX/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!T-GCN-WX/dense_8/Tensordot/MatMul
"T-GCN-WX/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"T-GCN-WX/dense_8/Tensordot/Const_2
(T-GCN-WX/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(T-GCN-WX/dense_8/Tensordot/concat_1/axis
#T-GCN-WX/dense_8/Tensordot/concat_1ConcatV2,T-GCN-WX/dense_8/Tensordot/GatherV2:output:0+T-GCN-WX/dense_8/Tensordot/Const_2:output:01T-GCN-WX/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#T-GCN-WX/dense_8/Tensordot/concat_1Ø
T-GCN-WX/dense_8/TensordotReshape+T-GCN-WX/dense_8/Tensordot/MatMul:product:0,T-GCN-WX/dense_8/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
T-GCN-WX/dense_8/Tensordot¿
'T-GCN-WX/dense_8/BiasAdd/ReadVariableOpReadVariableOp0t_gcn_wx_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'T-GCN-WX/dense_8/BiasAdd/ReadVariableOpÏ
T-GCN-WX/dense_8/BiasAddBiasAdd#T-GCN-WX/dense_8/Tensordot:output:0/T-GCN-WX/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
T-GCN-WX/dense_8/BiasAdd
T-GCN-WX/reshape_13/ShapeShape!T-GCN-WX/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:2
T-GCN-WX/reshape_13/Shape
'T-GCN-WX/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'T-GCN-WX/reshape_13/strided_slice/stack 
)T-GCN-WX/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)T-GCN-WX/reshape_13/strided_slice/stack_1 
)T-GCN-WX/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)T-GCN-WX/reshape_13/strided_slice/stack_2Ú
!T-GCN-WX/reshape_13/strided_sliceStridedSlice"T-GCN-WX/reshape_13/Shape:output:00T-GCN-WX/reshape_13/strided_slice/stack:output:02T-GCN-WX/reshape_13/strided_slice/stack_1:output:02T-GCN-WX/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!T-GCN-WX/reshape_13/strided_slice
#T-GCN-WX/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#T-GCN-WX/reshape_13/Reshape/shape/1
#T-GCN-WX/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#T-GCN-WX/reshape_13/Reshape/shape/2
!T-GCN-WX/reshape_13/Reshape/shapePack*T-GCN-WX/reshape_13/strided_slice:output:0,T-GCN-WX/reshape_13/Reshape/shape/1:output:0,T-GCN-WX/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!T-GCN-WX/reshape_13/Reshape/shapeÊ
T-GCN-WX/reshape_13/ReshapeReshape!T-GCN-WX/dense_8/BiasAdd:output:0*T-GCN-WX/reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
T-GCN-WX/reshape_13/Reshape¯
0T-GCN-WX/model_8/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0T-GCN-WX/model_8/tf.expand_dims_3/ExpandDims/dim
,T-GCN-WX/model_8/tf.expand_dims_3/ExpandDims
ExpandDims$T-GCN-WX/reshape_13/Reshape:output:09T-GCN-WX/model_8/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2.
,T-GCN-WX/model_8/tf.expand_dims_3/ExpandDims«
!T-GCN-WX/model_8/reshape_14/ShapeShape5T-GCN-WX/model_8/tf.expand_dims_3/ExpandDims:output:0*
T0*
_output_shapes
:2#
!T-GCN-WX/model_8/reshape_14/Shape¬
/T-GCN-WX/model_8/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_8/reshape_14/strided_slice/stack°
1T-GCN-WX/model_8/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_8/reshape_14/strided_slice/stack_1°
1T-GCN-WX/model_8/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_8/reshape_14/strided_slice/stack_2
)T-GCN-WX/model_8/reshape_14/strided_sliceStridedSlice*T-GCN-WX/model_8/reshape_14/Shape:output:08T-GCN-WX/model_8/reshape_14/strided_slice/stack:output:0:T-GCN-WX/model_8/reshape_14/strided_slice/stack_1:output:0:T-GCN-WX/model_8/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)T-GCN-WX/model_8/reshape_14/strided_slice
+T-GCN-WX/model_8/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2-
+T-GCN-WX/model_8/reshape_14/Reshape/shape/1
+T-GCN-WX/model_8/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+T-GCN-WX/model_8/reshape_14/Reshape/shape/2¬
)T-GCN-WX/model_8/reshape_14/Reshape/shapePack2T-GCN-WX/model_8/reshape_14/strided_slice:output:04T-GCN-WX/model_8/reshape_14/Reshape/shape/1:output:04T-GCN-WX/model_8/reshape_14/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)T-GCN-WX/model_8/reshape_14/Reshape/shapeö
#T-GCN-WX/model_8/reshape_14/ReshapeReshape5T-GCN-WX/model_8/tf.expand_dims_3/ExpandDims:output:02T-GCN-WX/model_8/reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2%
#T-GCN-WX/model_8/reshape_14/Reshapeß
CT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
CT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose/perm¿
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose	Transpose,T-GCN-WX/model_8/reshape_14/Reshape:output:0LT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transposeê
:T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/ShapeShapeBT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose:y:0*
T0*
_output_shapes
:2<
:T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shapeû
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstackUnpackCT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num2>
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack¯
KT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOpReadVariableOpTt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02M
KT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOpÍ
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2>
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_1ÿ
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack_1UnpackET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack_1Ù
BT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2D
BT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape/shapeÊ
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/ReshapeReshapeBT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose:y:0KT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2>
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape·
OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpReadVariableOpTt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02Q
OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpß
ET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2G
ET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/permã
@T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1	TransposeWT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp:value:0NT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/perm:output:0*
T0*
_output_shapes

:FF2B
@T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1Ý
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2F
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_1/shapeÉ
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_1ReshapeDT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1:y:0MT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_1Æ
;T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/MatMulMatMulET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape:output:0GT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2=
;T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/MatMulÒ
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2H
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1Ò
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2H
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2«
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shapePackET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack:output:0OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1:output:0OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2F
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape×
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2ReshapeET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/MatMul:product:0MT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2ã
ET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2G
ET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_2/permà
@T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_2	TransposeGT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_2:output:0NT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2B
@T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_2ð
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_2ShapeDT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_2:y:0*
T0*
_output_shapes
:2>
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_2
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack_2UnpackET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack_2¯
KT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOpReadVariableOpTt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02M
KT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOpÍ
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2>
<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_3ÿ
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack_3UnpackET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack_3Ý
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_3/shapeÒ
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_3ReshapeDT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_2:y:0MT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_3·
OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpReadVariableOpTt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02Q
OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpß
ET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2G
ET-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/permã
@T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3	TransposeWT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp:value:0NT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/perm:output:0*
T0*
_output_shapes

:
2B
@T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3Ý
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2F
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_4/shapeÉ
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_4ReshapeDT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3:y:0MT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_4/shape:output:0*
T0*
_output_shapes

:
2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_4Ì
=T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/MatMul_1MatMulGT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_3:output:0GT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2?
=T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/MatMul_1Ò
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2H
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1Ò
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2H
FT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2­
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shapePackGT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/unstack_2:output:0OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1:output:0OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2F
DT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shapeÙ
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5ReshapeGT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/MatMul_1:product:0MT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2@
>T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5£
GT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpReadVariableOpPt_gcn_wx_model_8_fixed_adjacency_graph_convolution_3_add_readvariableop_resource*
_output_shapes

:F*
dtype02I
GT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpÍ
8T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/addAddV2GT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/Reshape_5:output:0OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2:
8T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add²
!T-GCN-WX/model_8/reshape_15/ShapeShape<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add:z:0*
T0*
_output_shapes
:2#
!T-GCN-WX/model_8/reshape_15/Shape¬
/T-GCN-WX/model_8/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_8/reshape_15/strided_slice/stack°
1T-GCN-WX/model_8/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_8/reshape_15/strided_slice/stack_1°
1T-GCN-WX/model_8/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_8/reshape_15/strided_slice/stack_2
)T-GCN-WX/model_8/reshape_15/strided_sliceStridedSlice*T-GCN-WX/model_8/reshape_15/Shape:output:08T-GCN-WX/model_8/reshape_15/strided_slice/stack:output:0:T-GCN-WX/model_8/reshape_15/strided_slice/stack_1:output:0:T-GCN-WX/model_8/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)T-GCN-WX/model_8/reshape_15/strided_slice
+T-GCN-WX/model_8/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2-
+T-GCN-WX/model_8/reshape_15/Reshape/shape/1¥
+T-GCN-WX/model_8/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+T-GCN-WX/model_8/reshape_15/Reshape/shape/2
+T-GCN-WX/model_8/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+T-GCN-WX/model_8/reshape_15/Reshape/shape/3â
)T-GCN-WX/model_8/reshape_15/Reshape/shapePack2T-GCN-WX/model_8/reshape_15/strided_slice:output:04T-GCN-WX/model_8/reshape_15/Reshape/shape/1:output:04T-GCN-WX/model_8/reshape_15/Reshape/shape/2:output:04T-GCN-WX/model_8/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)T-GCN-WX/model_8/reshape_15/Reshape/shape
#T-GCN-WX/model_8/reshape_15/ReshapeReshape<T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add:z:02T-GCN-WX/model_8/reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2%
#T-GCN-WX/model_8/reshape_15/Reshape¯
)T-GCN-WX/model_8/permute_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)T-GCN-WX/model_8/permute_3/transpose/permõ
$T-GCN-WX/model_8/permute_3/transpose	Transpose,T-GCN-WX/model_8/reshape_15/Reshape:output:02T-GCN-WX/model_8/permute_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2&
$T-GCN-WX/model_8/permute_3/transpose
!T-GCN-WX/model_8/reshape_16/ShapeShape(T-GCN-WX/model_8/permute_3/transpose:y:0*
T0*
_output_shapes
:2#
!T-GCN-WX/model_8/reshape_16/Shape¬
/T-GCN-WX/model_8/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_8/reshape_16/strided_slice/stack°
1T-GCN-WX/model_8/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_8/reshape_16/strided_slice/stack_1°
1T-GCN-WX/model_8/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_8/reshape_16/strided_slice/stack_2
)T-GCN-WX/model_8/reshape_16/strided_sliceStridedSlice*T-GCN-WX/model_8/reshape_16/Shape:output:08T-GCN-WX/model_8/reshape_16/strided_slice/stack:output:0:T-GCN-WX/model_8/reshape_16/strided_slice/stack_1:output:0:T-GCN-WX/model_8/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)T-GCN-WX/model_8/reshape_16/strided_slice¥
+T-GCN-WX/model_8/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+T-GCN-WX/model_8/reshape_16/Reshape/shape/1
+T-GCN-WX/model_8/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2-
+T-GCN-WX/model_8/reshape_16/Reshape/shape/2¬
)T-GCN-WX/model_8/reshape_16/Reshape/shapePack2T-GCN-WX/model_8/reshape_16/strided_slice:output:04T-GCN-WX/model_8/reshape_16/Reshape/shape/1:output:04T-GCN-WX/model_8/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)T-GCN-WX/model_8/reshape_16/Reshape/shapeé
#T-GCN-WX/model_8/reshape_16/ReshapeReshape(T-GCN-WX/model_8/permute_3/transpose:y:02T-GCN-WX/model_8/reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2%
#T-GCN-WX/model_8/reshape_16/Reshape
T-GCN-WX/model_8/lstm_3/ShapeShape,T-GCN-WX/model_8/reshape_16/Reshape:output:0*
T0*
_output_shapes
:2
T-GCN-WX/model_8/lstm_3/Shape¤
+T-GCN-WX/model_8/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+T-GCN-WX/model_8/lstm_3/strided_slice/stack¨
-T-GCN-WX/model_8/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-T-GCN-WX/model_8/lstm_3/strided_slice/stack_1¨
-T-GCN-WX/model_8/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-T-GCN-WX/model_8/lstm_3/strided_slice/stack_2ò
%T-GCN-WX/model_8/lstm_3/strided_sliceStridedSlice&T-GCN-WX/model_8/lstm_3/Shape:output:04T-GCN-WX/model_8/lstm_3/strided_slice/stack:output:06T-GCN-WX/model_8/lstm_3/strided_slice/stack_1:output:06T-GCN-WX/model_8/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%T-GCN-WX/model_8/lstm_3/strided_slice
#T-GCN-WX/model_8/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2%
#T-GCN-WX/model_8/lstm_3/zeros/mul/yÌ
!T-GCN-WX/model_8/lstm_3/zeros/mulMul.T-GCN-WX/model_8/lstm_3/strided_slice:output:0,T-GCN-WX/model_8/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!T-GCN-WX/model_8/lstm_3/zeros/mul
$T-GCN-WX/model_8/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$T-GCN-WX/model_8/lstm_3/zeros/Less/yÇ
"T-GCN-WX/model_8/lstm_3/zeros/LessLess%T-GCN-WX/model_8/lstm_3/zeros/mul:z:0-T-GCN-WX/model_8/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"T-GCN-WX/model_8/lstm_3/zeros/Less
&T-GCN-WX/model_8/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2(
&T-GCN-WX/model_8/lstm_3/zeros/packed/1ã
$T-GCN-WX/model_8/lstm_3/zeros/packedPack.T-GCN-WX/model_8/lstm_3/strided_slice:output:0/T-GCN-WX/model_8/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$T-GCN-WX/model_8/lstm_3/zeros/packed
#T-GCN-WX/model_8/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#T-GCN-WX/model_8/lstm_3/zeros/ConstÖ
T-GCN-WX/model_8/lstm_3/zerosFill-T-GCN-WX/model_8/lstm_3/zeros/packed:output:0,T-GCN-WX/model_8/lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
T-GCN-WX/model_8/lstm_3/zeros
%T-GCN-WX/model_8/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2'
%T-GCN-WX/model_8/lstm_3/zeros_1/mul/yÒ
#T-GCN-WX/model_8/lstm_3/zeros_1/mulMul.T-GCN-WX/model_8/lstm_3/strided_slice:output:0.T-GCN-WX/model_8/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#T-GCN-WX/model_8/lstm_3/zeros_1/mul
&T-GCN-WX/model_8/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2(
&T-GCN-WX/model_8/lstm_3/zeros_1/Less/yÏ
$T-GCN-WX/model_8/lstm_3/zeros_1/LessLess'T-GCN-WX/model_8/lstm_3/zeros_1/mul:z:0/T-GCN-WX/model_8/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2&
$T-GCN-WX/model_8/lstm_3/zeros_1/Less
(T-GCN-WX/model_8/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2*
(T-GCN-WX/model_8/lstm_3/zeros_1/packed/1é
&T-GCN-WX/model_8/lstm_3/zeros_1/packedPack.T-GCN-WX/model_8/lstm_3/strided_slice:output:01T-GCN-WX/model_8/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&T-GCN-WX/model_8/lstm_3/zeros_1/packed
%T-GCN-WX/model_8/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%T-GCN-WX/model_8/lstm_3/zeros_1/ConstÞ
T-GCN-WX/model_8/lstm_3/zeros_1Fill/T-GCN-WX/model_8/lstm_3/zeros_1/packed:output:0.T-GCN-WX/model_8/lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
T-GCN-WX/model_8/lstm_3/zeros_1¥
&T-GCN-WX/model_8/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&T-GCN-WX/model_8/lstm_3/transpose/permè
!T-GCN-WX/model_8/lstm_3/transpose	Transpose,T-GCN-WX/model_8/reshape_16/Reshape:output:0/T-GCN-WX/model_8/lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2#
!T-GCN-WX/model_8/lstm_3/transpose
T-GCN-WX/model_8/lstm_3/Shape_1Shape%T-GCN-WX/model_8/lstm_3/transpose:y:0*
T0*
_output_shapes
:2!
T-GCN-WX/model_8/lstm_3/Shape_1¨
-T-GCN-WX/model_8/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-T-GCN-WX/model_8/lstm_3/strided_slice_1/stack¬
/T-GCN-WX/model_8/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_8/lstm_3/strided_slice_1/stack_1¬
/T-GCN-WX/model_8/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_8/lstm_3/strided_slice_1/stack_2þ
'T-GCN-WX/model_8/lstm_3/strided_slice_1StridedSlice(T-GCN-WX/model_8/lstm_3/Shape_1:output:06T-GCN-WX/model_8/lstm_3/strided_slice_1/stack:output:08T-GCN-WX/model_8/lstm_3/strided_slice_1/stack_1:output:08T-GCN-WX/model_8/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'T-GCN-WX/model_8/lstm_3/strided_slice_1µ
3T-GCN-WX/model_8/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ25
3T-GCN-WX/model_8/lstm_3/TensorArrayV2/element_shape
%T-GCN-WX/model_8/lstm_3/TensorArrayV2TensorListReserve<T-GCN-WX/model_8/lstm_3/TensorArrayV2/element_shape:output:00T-GCN-WX/model_8/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%T-GCN-WX/model_8/lstm_3/TensorArrayV2ï
MT-GCN-WX/model_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2O
MT-GCN-WX/model_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeØ
?T-GCN-WX/model_8/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%T-GCN-WX/model_8/lstm_3/transpose:y:0VT-GCN-WX/model_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?T-GCN-WX/model_8/lstm_3/TensorArrayUnstack/TensorListFromTensor¨
-T-GCN-WX/model_8/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-T-GCN-WX/model_8/lstm_3/strided_slice_2/stack¬
/T-GCN-WX/model_8/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_8/lstm_3/strided_slice_2/stack_1¬
/T-GCN-WX/model_8/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_8/lstm_3/strided_slice_2/stack_2
'T-GCN-WX/model_8/lstm_3/strided_slice_2StridedSlice%T-GCN-WX/model_8/lstm_3/transpose:y:06T-GCN-WX/model_8/lstm_3/strided_slice_2/stack:output:08T-GCN-WX/model_8/lstm_3/strided_slice_2/stack_1:output:08T-GCN-WX/model_8/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2)
'T-GCN-WX/model_8/lstm_3/strided_slice_2ú
9T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOpBt_gcn_wx_model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02;
9T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp
*T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMulMatMul0T-GCN-WX/model_8/lstm_3/strided_slice_2:output:0AT-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul
;T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOpDt_gcn_wx_model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02=
;T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp
,T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1MatMul&T-GCN-WX/model_8/lstm_3/zeros:output:0CT-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1ü
'T-GCN-WX/model_8/lstm_3/lstm_cell_3/addAddV24T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul:product:06T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'T-GCN-WX/model_8/lstm_3/lstm_cell_3/addù
:T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOpCt_gcn_wx_model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02<
:T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp
+T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAddBiasAdd+T-GCN-WX/model_8/lstm_3/lstm_cell_3/add:z:0BT-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/Const¬
3T-GCN-WX/model_8/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3T-GCN-WX/model_8/lstm_3/lstm_cell_3/split/split_dimÓ
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/splitSplit<T-GCN-WX/model_8/lstm_3/lstm_cell_3/split/split_dim:output:04T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2+
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/splitÌ
+T-GCN-WX/model_8/lstm_3/lstm_cell_3/SigmoidSigmoid2T-GCN-WX/model_8/lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2-
+T-GCN-WX/model_8/lstm_3/lstm_cell_3/SigmoidÐ
-T-GCN-WX/model_8/lstm_3/lstm_cell_3/Sigmoid_1Sigmoid2T-GCN-WX/model_8/lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2/
-T-GCN-WX/model_8/lstm_3/lstm_cell_3/Sigmoid_1é
'T-GCN-WX/model_8/lstm_3/lstm_cell_3/mulMul1T-GCN-WX/model_8/lstm_3/lstm_cell_3/Sigmoid_1:y:0(T-GCN-WX/model_8/lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'T-GCN-WX/model_8/lstm_3/lstm_cell_3/mulõ
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/mul_1Mul/T-GCN-WX/model_8/lstm_3/lstm_cell_3/Sigmoid:y:02T-GCN-WX/model_8/lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/mul_1î
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/add_1AddV2+T-GCN-WX/model_8/lstm_3/lstm_cell_3/mul:z:0-T-GCN-WX/model_8/lstm_3/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/add_1Ð
-T-GCN-WX/model_8/lstm_3/lstm_cell_3/Sigmoid_2Sigmoid2T-GCN-WX/model_8/lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2/
-T-GCN-WX/model_8/lstm_3/lstm_cell_3/Sigmoid_2ò
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/mul_2Mul1T-GCN-WX/model_8/lstm_3/lstm_cell_3/Sigmoid_2:y:0-T-GCN-WX/model_8/lstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)T-GCN-WX/model_8/lstm_3/lstm_cell_3/mul_2¿
5T-GCN-WX/model_8/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5T-GCN-WX/model_8/lstm_3/TensorArrayV2_1/element_shape
'T-GCN-WX/model_8/lstm_3/TensorArrayV2_1TensorListReserve>T-GCN-WX/model_8/lstm_3/TensorArrayV2_1/element_shape:output:00T-GCN-WX/model_8/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'T-GCN-WX/model_8/lstm_3/TensorArrayV2_1~
T-GCN-WX/model_8/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
T-GCN-WX/model_8/lstm_3/time¯
0T-GCN-WX/model_8/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0T-GCN-WX/model_8/lstm_3/while/maximum_iterations
*T-GCN-WX/model_8/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*T-GCN-WX/model_8/lstm_3/while/loop_counterÕ
T-GCN-WX/model_8/lstm_3/whileWhile3T-GCN-WX/model_8/lstm_3/while/loop_counter:output:09T-GCN-WX/model_8/lstm_3/while/maximum_iterations:output:0%T-GCN-WX/model_8/lstm_3/time:output:00T-GCN-WX/model_8/lstm_3/TensorArrayV2_1:handle:0&T-GCN-WX/model_8/lstm_3/zeros:output:0(T-GCN-WX/model_8/lstm_3/zeros_1:output:00T-GCN-WX/model_8/lstm_3/strided_slice_1:output:0OT-GCN-WX/model_8/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bt_gcn_wx_model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resourceDt_gcn_wx_model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resourceCt_gcn_wx_model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*4
body,R*
(T-GCN-WX_model_8_lstm_3_while_body_58508*4
cond,R*
(T-GCN-WX_model_8_lstm_3_while_cond_58507*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
T-GCN-WX/model_8/lstm_3/whileå
HT-GCN-WX/model_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2J
HT-GCN-WX/model_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeÉ
:T-GCN-WX/model_8/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStack&T-GCN-WX/model_8/lstm_3/while:output:3QT-GCN-WX/model_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02<
:T-GCN-WX/model_8/lstm_3/TensorArrayV2Stack/TensorListStack±
-T-GCN-WX/model_8/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2/
-T-GCN-WX/model_8/lstm_3/strided_slice_3/stack¬
/T-GCN-WX/model_8/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_8/lstm_3/strided_slice_3/stack_1¬
/T-GCN-WX/model_8/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_8/lstm_3/strided_slice_3/stack_2«
'T-GCN-WX/model_8/lstm_3/strided_slice_3StridedSliceCT-GCN-WX/model_8/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:06T-GCN-WX/model_8/lstm_3/strided_slice_3/stack:output:08T-GCN-WX/model_8/lstm_3/strided_slice_3/stack_1:output:08T-GCN-WX/model_8/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2)
'T-GCN-WX/model_8/lstm_3/strided_slice_3©
(T-GCN-WX/model_8/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(T-GCN-WX/model_8/lstm_3/transpose_1/perm
#T-GCN-WX/model_8/lstm_3/transpose_1	TransposeCT-GCN-WX/model_8/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:01T-GCN-WX/model_8/lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2%
#T-GCN-WX/model_8/lstm_3/transpose_1
T-GCN-WX/model_8/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2!
T-GCN-WX/model_8/lstm_3/runtime»
#T-GCN-WX/model_8/dropout_5/IdentityIdentity0T-GCN-WX/model_8/lstm_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#T-GCN-WX/model_8/dropout_5/IdentityÙ
.T-GCN-WX/model_8/dense_7/MatMul/ReadVariableOpReadVariableOp7t_gcn_wx_model_8_dense_7_matmul_readvariableop_resource*
_output_shapes
:	ÈF*
dtype020
.T-GCN-WX/model_8/dense_7/MatMul/ReadVariableOpä
T-GCN-WX/model_8/dense_7/MatMulMatMul,T-GCN-WX/model_8/dropout_5/Identity:output:06T-GCN-WX/model_8/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2!
T-GCN-WX/model_8/dense_7/MatMul×
/T-GCN-WX/model_8/dense_7/BiasAdd/ReadVariableOpReadVariableOp8t_gcn_wx_model_8_dense_7_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype021
/T-GCN-WX/model_8/dense_7/BiasAdd/ReadVariableOpå
 T-GCN-WX/model_8/dense_7/BiasAddBiasAdd)T-GCN-WX/model_8/dense_7/MatMul:product:07T-GCN-WX/model_8/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2"
 T-GCN-WX/model_8/dense_7/BiasAdd¬
 T-GCN-WX/model_8/dense_7/SigmoidSigmoid)T-GCN-WX/model_8/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2"
 T-GCN-WX/model_8/dense_7/Sigmoidö
IdentityIdentity$T-GCN-WX/model_8/dense_7/Sigmoid:y:0(^T-GCN-WX/dense_8/BiasAdd/ReadVariableOp*^T-GCN-WX/dense_8/Tensordot/ReadVariableOp0^T-GCN-WX/model_8/dense_7/BiasAdd/ReadVariableOp/^T-GCN-WX/model_8/dense_7/MatMul/ReadVariableOpH^T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpP^T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpP^T-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp;^T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:^T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp<^T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^T-GCN-WX/model_8/lstm_3/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::2R
'T-GCN-WX/dense_8/BiasAdd/ReadVariableOp'T-GCN-WX/dense_8/BiasAdd/ReadVariableOp2V
)T-GCN-WX/dense_8/Tensordot/ReadVariableOp)T-GCN-WX/dense_8/Tensordot/ReadVariableOp2b
/T-GCN-WX/model_8/dense_7/BiasAdd/ReadVariableOp/T-GCN-WX/model_8/dense_7/BiasAdd/ReadVariableOp2`
.T-GCN-WX/model_8/dense_7/MatMul/ReadVariableOp.T-GCN-WX/model_8/dense_7/MatMul/ReadVariableOp2
GT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpGT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp2¢
OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpOT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp2¢
OT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpOT-GCN-WX/model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp2x
:T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:T-GCN-WX/model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2v
9T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp9T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp2z
;T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp;T-GCN-WX/model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2>
T-GCN-WX/model_8/lstm_3/whileT-GCN-WX/model_8/lstm_3/while:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_11
ü

(T-GCN-WX_model_8_lstm_3_while_cond_58507L
Ht_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_loop_counterR
Nt_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_maximum_iterations-
)t_gcn_wx_model_8_lstm_3_while_placeholder/
+t_gcn_wx_model_8_lstm_3_while_placeholder_1/
+t_gcn_wx_model_8_lstm_3_while_placeholder_2/
+t_gcn_wx_model_8_lstm_3_while_placeholder_3N
Jt_gcn_wx_model_8_lstm_3_while_less_t_gcn_wx_model_8_lstm_3_strided_slice_1c
_t_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_cond_58507___redundant_placeholder0c
_t_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_cond_58507___redundant_placeholder1c
_t_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_cond_58507___redundant_placeholder2c
_t_gcn_wx_model_8_lstm_3_while_t_gcn_wx_model_8_lstm_3_while_cond_58507___redundant_placeholder3*
&t_gcn_wx_model_8_lstm_3_while_identity
è
"T-GCN-WX/model_8/lstm_3/while/LessLess)t_gcn_wx_model_8_lstm_3_while_placeholderJt_gcn_wx_model_8_lstm_3_while_less_t_gcn_wx_model_8_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2$
"T-GCN-WX/model_8/lstm_3/while/Less¥
&T-GCN-WX/model_8/lstm_3/while/IdentityIdentity&T-GCN-WX/model_8/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2(
&T-GCN-WX/model_8/lstm_3/while/Identity"Y
&t_gcn_wx_model_8_lstm_3_while_identity/T-GCN-WX/model_8/lstm_3/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
¶Â
ñ	
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60458

inputs-
)dense_8_tensordot_readvariableop_resource+
'dense_8_biasadd_readvariableop_resourceO
Kmodel_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resourceO
Kmodel_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resourceK
Gmodel_8_fixed_adjacency_graph_convolution_3_add_readvariableop_resource=
9model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resource?
;model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource>
:model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource2
.model_8_dense_7_matmul_readvariableop_resource3
/model_8_dense_7_biasadd_readvariableop_resource
identity¢dense_8/BiasAdd/ReadVariableOp¢ dense_8/Tensordot/ReadVariableOp¢&model_8/dense_7/BiasAdd/ReadVariableOp¢%model_8/dense_7/MatMul/ReadVariableOp¢>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp¢Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp¢Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp¢1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp¢0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp¢2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp¢model_8/lstm_3/while®
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axes
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_8/Tensordot/freeh
dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_8/Tensordot/Shape
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axisù
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axisÿ
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1¨
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axisØ
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat¬
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stack¬
dense_8/Tensordot/transpose	Transposeinputs!dense_8/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_8/Tensordot/transpose¿
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_8/Tensordot/Reshape¾
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/Tensordot/MatMul
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/Const_2
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axiså
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1´
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_8/Tensordot¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp«
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_8/BiasAddl
reshape_13/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_13/Shape
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2¤
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_13/Reshape/shape/1z
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/2×
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape¦
reshape_13/ReshapeReshapedense_8/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_13/Reshape
'model_8/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_8/tf.expand_dims_3/ExpandDims/dimá
#model_8/tf.expand_dims_3/ExpandDims
ExpandDimsreshape_13/Reshape:output:00model_8/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2%
#model_8/tf.expand_dims_3/ExpandDims
model_8/reshape_14/ShapeShape,model_8/tf.expand_dims_3/ExpandDims:output:0*
T0*
_output_shapes
:2
model_8/reshape_14/Shape
&model_8/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_14/strided_slice/stack
(model_8/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_14/strided_slice/stack_1
(model_8/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_14/strided_slice/stack_2Ô
 model_8/reshape_14/strided_sliceStridedSlice!model_8/reshape_14/Shape:output:0/model_8/reshape_14/strided_slice/stack:output:01model_8/reshape_14/strided_slice/stack_1:output:01model_8/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_14/strided_slice
"model_8/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_14/Reshape/shape/1
"model_8/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_8/reshape_14/Reshape/shape/2ÿ
 model_8/reshape_14/Reshape/shapePack)model_8/reshape_14/strided_slice:output:0+model_8/reshape_14/Reshape/shape/1:output:0+model_8/reshape_14/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_14/Reshape/shapeÒ
model_8/reshape_14/ReshapeReshape,model_8/tf.expand_dims_3/ExpandDims:output:0)model_8/reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/reshape_14/ReshapeÍ
:model_8/fixed_adjacency_graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2<
:model_8/fixed_adjacency_graph_convolution_3/transpose/perm
5model_8/fixed_adjacency_graph_convolution_3/transpose	Transpose#model_8/reshape_14/Reshape:output:0Cmodel_8/fixed_adjacency_graph_convolution_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF27
5model_8/fixed_adjacency_graph_convolution_3/transposeÏ
1model_8/fixed_adjacency_graph_convolution_3/ShapeShape9model_8/fixed_adjacency_graph_convolution_3/transpose:y:0*
T0*
_output_shapes
:23
1model_8/fixed_adjacency_graph_convolution_3/Shapeà
3model_8/fixed_adjacency_graph_convolution_3/unstackUnpack:model_8/fixed_adjacency_graph_convolution_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num25
3model_8/fixed_adjacency_graph_convolution_3/unstack
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02D
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOp»
3model_8/fixed_adjacency_graph_convolution_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   25
3model_8/fixed_adjacency_graph_convolution_3/Shape_1ä
5model_8/fixed_adjacency_graph_convolution_3/unstack_1Unpack<model_8/fixed_adjacency_graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num27
5model_8/fixed_adjacency_graph_convolution_3/unstack_1Ç
9model_8/fixed_adjacency_graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2;
9model_8/fixed_adjacency_graph_convolution_3/Reshape/shape¦
3model_8/fixed_adjacency_graph_convolution_3/ReshapeReshape9model_8/fixed_adjacency_graph_convolution_3/transpose:y:0Bmodel_8/fixed_adjacency_graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF25
3model_8/fixed_adjacency_graph_convolution_3/Reshape
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02H
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpÍ
<model_8/fixed_adjacency_graph_convolution_3/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_8/fixed_adjacency_graph_convolution_3/transpose_1/perm¿
7model_8/fixed_adjacency_graph_convolution_3/transpose_1	TransposeNmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp:value:0Emodel_8/fixed_adjacency_graph_convolution_3/transpose_1/perm:output:0*
T0*
_output_shapes

:FF29
7model_8/fixed_adjacency_graph_convolution_3/transpose_1Ë
;model_8/fixed_adjacency_graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_1/shape¥
5model_8/fixed_adjacency_graph_convolution_3/Reshape_1Reshape;model_8/fixed_adjacency_graph_convolution_3/transpose_1:y:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_1¢
2model_8/fixed_adjacency_graph_convolution_3/MatMulMatMul<model_8/fixed_adjacency_graph_convolution_3/Reshape:output:0>model_8/fixed_adjacency_graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF24
2model_8/fixed_adjacency_graph_convolution_3/MatMulÀ
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1À
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2þ
;model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shapePack<model_8/fixed_adjacency_graph_convolution_3/unstack:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape³
5model_8/fixed_adjacency_graph_convolution_3/Reshape_2Reshape<model_8/fixed_adjacency_graph_convolution_3/MatMul:product:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_2Ñ
<model_8/fixed_adjacency_graph_convolution_3/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<model_8/fixed_adjacency_graph_convolution_3/transpose_2/perm¼
7model_8/fixed_adjacency_graph_convolution_3/transpose_2	Transpose>model_8/fixed_adjacency_graph_convolution_3/Reshape_2:output:0Emodel_8/fixed_adjacency_graph_convolution_3/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF29
7model_8/fixed_adjacency_graph_convolution_3/transpose_2Õ
3model_8/fixed_adjacency_graph_convolution_3/Shape_2Shape;model_8/fixed_adjacency_graph_convolution_3/transpose_2:y:0*
T0*
_output_shapes
:25
3model_8/fixed_adjacency_graph_convolution_3/Shape_2æ
5model_8/fixed_adjacency_graph_convolution_3/unstack_2Unpack<model_8/fixed_adjacency_graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num27
5model_8/fixed_adjacency_graph_convolution_3/unstack_2
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02D
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOp»
3model_8/fixed_adjacency_graph_convolution_3/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   25
3model_8/fixed_adjacency_graph_convolution_3/Shape_3ä
5model_8/fixed_adjacency_graph_convolution_3/unstack_3Unpack<model_8/fixed_adjacency_graph_convolution_3/Shape_3:output:0*
T0*
_output_shapes
: : *	
num27
5model_8/fixed_adjacency_graph_convolution_3/unstack_3Ë
;model_8/fixed_adjacency_graph_convolution_3/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_3/shape®
5model_8/fixed_adjacency_graph_convolution_3/Reshape_3Reshape;model_8/fixed_adjacency_graph_convolution_3/transpose_2:y:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_3
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02H
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpÍ
<model_8/fixed_adjacency_graph_convolution_3/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_8/fixed_adjacency_graph_convolution_3/transpose_3/perm¿
7model_8/fixed_adjacency_graph_convolution_3/transpose_3	TransposeNmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp:value:0Emodel_8/fixed_adjacency_graph_convolution_3/transpose_3/perm:output:0*
T0*
_output_shapes

:
29
7model_8/fixed_adjacency_graph_convolution_3/transpose_3Ë
;model_8/fixed_adjacency_graph_convolution_3/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_4/shape¥
5model_8/fixed_adjacency_graph_convolution_3/Reshape_4Reshape;model_8/fixed_adjacency_graph_convolution_3/transpose_3:y:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_4/shape:output:0*
T0*
_output_shapes

:
27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_4¨
4model_8/fixed_adjacency_graph_convolution_3/MatMul_1MatMul>model_8/fixed_adjacency_graph_convolution_3/Reshape_3:output:0>model_8/fixed_adjacency_graph_convolution_3/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
26
4model_8/fixed_adjacency_graph_convolution_3/MatMul_1À
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1À
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2
;model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shapePack>model_8/fixed_adjacency_graph_convolution_3/unstack_2:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shapeµ
5model_8/fixed_adjacency_graph_convolution_3/Reshape_5Reshape>model_8/fixed_adjacency_graph_convolution_3/MatMul_1:product:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_5
>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpReadVariableOpGmodel_8_fixed_adjacency_graph_convolution_3_add_readvariableop_resource*
_output_shapes

:F*
dtype02@
>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp©
/model_8/fixed_adjacency_graph_convolution_3/addAddV2>model_8/fixed_adjacency_graph_convolution_3/Reshape_5:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
21
/model_8/fixed_adjacency_graph_convolution_3/add
model_8/reshape_15/ShapeShape3model_8/fixed_adjacency_graph_convolution_3/add:z:0*
T0*
_output_shapes
:2
model_8/reshape_15/Shape
&model_8/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_15/strided_slice/stack
(model_8/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_15/strided_slice/stack_1
(model_8/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_15/strided_slice/stack_2Ô
 model_8/reshape_15/strided_sliceStridedSlice!model_8/reshape_15/Shape:output:0/model_8/reshape_15/strided_slice/stack:output:01model_8/reshape_15/strided_slice/stack_1:output:01model_8/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_15/strided_slice
"model_8/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_15/Reshape/shape/1
"model_8/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"model_8/reshape_15/Reshape/shape/2
"model_8/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_8/reshape_15/Reshape/shape/3¬
 model_8/reshape_15/Reshape/shapePack)model_8/reshape_15/strided_slice:output:0+model_8/reshape_15/Reshape/shape/1:output:0+model_8/reshape_15/Reshape/shape/2:output:0+model_8/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_15/Reshape/shapeÝ
model_8/reshape_15/ReshapeReshape3model_8/fixed_adjacency_graph_convolution_3/add:z:0)model_8/reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
model_8/reshape_15/Reshape
 model_8/permute_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 model_8/permute_3/transpose/permÑ
model_8/permute_3/transpose	Transpose#model_8/reshape_15/Reshape:output:0)model_8/permute_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
model_8/permute_3/transpose
model_8/reshape_16/ShapeShapemodel_8/permute_3/transpose:y:0*
T0*
_output_shapes
:2
model_8/reshape_16/Shape
&model_8/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_16/strided_slice/stack
(model_8/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_16/strided_slice/stack_1
(model_8/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_16/strided_slice/stack_2Ô
 model_8/reshape_16/strided_sliceStridedSlice!model_8/reshape_16/Shape:output:0/model_8/reshape_16/strided_slice/stack:output:01model_8/reshape_16/strided_slice/stack_1:output:01model_8/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_16/strided_slice
"model_8/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"model_8/reshape_16/Reshape/shape/1
"model_8/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_16/Reshape/shape/2ÿ
 model_8/reshape_16/Reshape/shapePack)model_8/reshape_16/strided_slice:output:0+model_8/reshape_16/Reshape/shape/1:output:0+model_8/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_16/Reshape/shapeÅ
model_8/reshape_16/ReshapeReshapemodel_8/permute_3/transpose:y:0)model_8/reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
model_8/reshape_16/Reshape
model_8/lstm_3/ShapeShape#model_8/reshape_16/Reshape:output:0*
T0*
_output_shapes
:2
model_8/lstm_3/Shape
"model_8/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_8/lstm_3/strided_slice/stack
$model_8/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_8/lstm_3/strided_slice/stack_1
$model_8/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_8/lstm_3/strided_slice/stack_2¼
model_8/lstm_3/strided_sliceStridedSlicemodel_8/lstm_3/Shape:output:0+model_8/lstm_3/strided_slice/stack:output:0-model_8/lstm_3/strided_slice/stack_1:output:0-model_8/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_8/lstm_3/strided_slice{
model_8/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
model_8/lstm_3/zeros/mul/y¨
model_8/lstm_3/zeros/mulMul%model_8/lstm_3/strided_slice:output:0#model_8/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros/mul}
model_8/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
model_8/lstm_3/zeros/Less/y£
model_8/lstm_3/zeros/LessLessmodel_8/lstm_3/zeros/mul:z:0$model_8/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros/Less
model_8/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
model_8/lstm_3/zeros/packed/1¿
model_8/lstm_3/zeros/packedPack%model_8/lstm_3/strided_slice:output:0&model_8/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_8/lstm_3/zeros/packed}
model_8/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_3/zeros/Const²
model_8/lstm_3/zerosFill$model_8/lstm_3/zeros/packed:output:0#model_8/lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
model_8/lstm_3/zeros
model_8/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
model_8/lstm_3/zeros_1/mul/y®
model_8/lstm_3/zeros_1/mulMul%model_8/lstm_3/strided_slice:output:0%model_8/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros_1/mul
model_8/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
model_8/lstm_3/zeros_1/Less/y«
model_8/lstm_3/zeros_1/LessLessmodel_8/lstm_3/zeros_1/mul:z:0&model_8/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros_1/Less
model_8/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2!
model_8/lstm_3/zeros_1/packed/1Å
model_8/lstm_3/zeros_1/packedPack%model_8/lstm_3/strided_slice:output:0(model_8/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_8/lstm_3/zeros_1/packed
model_8/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_3/zeros_1/Constº
model_8/lstm_3/zeros_1Fill&model_8/lstm_3/zeros_1/packed:output:0%model_8/lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
model_8/lstm_3/zeros_1
model_8/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_8/lstm_3/transpose/permÄ
model_8/lstm_3/transpose	Transpose#model_8/reshape_16/Reshape:output:0&model_8/lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
model_8/lstm_3/transpose|
model_8/lstm_3/Shape_1Shapemodel_8/lstm_3/transpose:y:0*
T0*
_output_shapes
:2
model_8/lstm_3/Shape_1
$model_8/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_8/lstm_3/strided_slice_1/stack
&model_8/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_1/stack_1
&model_8/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_1/stack_2È
model_8/lstm_3/strided_slice_1StridedSlicemodel_8/lstm_3/Shape_1:output:0-model_8/lstm_3/strided_slice_1/stack:output:0/model_8/lstm_3/strided_slice_1/stack_1:output:0/model_8/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
model_8/lstm_3/strided_slice_1£
*model_8/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*model_8/lstm_3/TensorArrayV2/element_shapeî
model_8/lstm_3/TensorArrayV2TensorListReserve3model_8/lstm_3/TensorArrayV2/element_shape:output:0'model_8/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_8/lstm_3/TensorArrayV2Ý
Dmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2F
Dmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape´
6model_8/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_8/lstm_3/transpose:y:0Mmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6model_8/lstm_3/TensorArrayUnstack/TensorListFromTensor
$model_8/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_8/lstm_3/strided_slice_2/stack
&model_8/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_2/stack_1
&model_8/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_2/stack_2Ö
model_8/lstm_3/strided_slice_2StridedSlicemodel_8/lstm_3/transpose:y:0-model_8/lstm_3/strided_slice_2/stack:output:0/model_8/lstm_3/strided_slice_2/stack_1:output:0/model_8/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2 
model_8/lstm_3/strided_slice_2ß
0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype022
0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOpæ
!model_8/lstm_3/lstm_cell_3/MatMulMatMul'model_8/lstm_3/strided_slice_2:output:08model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!model_8/lstm_3/lstm_cell_3/MatMulæ
2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype024
2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpâ
#model_8/lstm_3/lstm_cell_3/MatMul_1MatMulmodel_8/lstm_3/zeros:output:0:model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#model_8/lstm_3/lstm_cell_3/MatMul_1Ø
model_8/lstm_3/lstm_cell_3/addAddV2+model_8/lstm_3/lstm_cell_3/MatMul:product:0-model_8/lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
model_8/lstm_3/lstm_cell_3/addÞ
1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype023
1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpå
"model_8/lstm_3/lstm_cell_3/BiasAddBiasAdd"model_8/lstm_3/lstm_cell_3/add:z:09model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"model_8/lstm_3/lstm_cell_3/BiasAdd
 model_8/lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_8/lstm_3/lstm_cell_3/Const
*model_8/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_8/lstm_3/lstm_cell_3/split/split_dim¯
 model_8/lstm_3/lstm_cell_3/splitSplit3model_8/lstm_3/lstm_cell_3/split/split_dim:output:0+model_8/lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2"
 model_8/lstm_3/lstm_cell_3/split±
"model_8/lstm_3/lstm_cell_3/SigmoidSigmoid)model_8/lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"model_8/lstm_3/lstm_cell_3/Sigmoidµ
$model_8/lstm_3/lstm_cell_3/Sigmoid_1Sigmoid)model_8/lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$model_8/lstm_3/lstm_cell_3/Sigmoid_1Å
model_8/lstm_3/lstm_cell_3/mulMul(model_8/lstm_3/lstm_cell_3/Sigmoid_1:y:0model_8/lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
model_8/lstm_3/lstm_cell_3/mulÑ
 model_8/lstm_3/lstm_cell_3/mul_1Mul&model_8/lstm_3/lstm_cell_3/Sigmoid:y:0)model_8/lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 model_8/lstm_3/lstm_cell_3/mul_1Ê
 model_8/lstm_3/lstm_cell_3/add_1AddV2"model_8/lstm_3/lstm_cell_3/mul:z:0$model_8/lstm_3/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 model_8/lstm_3/lstm_cell_3/add_1µ
$model_8/lstm_3/lstm_cell_3/Sigmoid_2Sigmoid)model_8/lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$model_8/lstm_3/lstm_cell_3/Sigmoid_2Î
 model_8/lstm_3/lstm_cell_3/mul_2Mul(model_8/lstm_3/lstm_cell_3/Sigmoid_2:y:0$model_8/lstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 model_8/lstm_3/lstm_cell_3/mul_2­
,model_8/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2.
,model_8/lstm_3/TensorArrayV2_1/element_shapeô
model_8/lstm_3/TensorArrayV2_1TensorListReserve5model_8/lstm_3/TensorArrayV2_1/element_shape:output:0'model_8/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_8/lstm_3/TensorArrayV2_1l
model_8/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_8/lstm_3/time
'model_8/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_8/lstm_3/while/maximum_iterations
!model_8/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model_8/lstm_3/while/loop_counterÎ
model_8/lstm_3/whileWhile*model_8/lstm_3/while/loop_counter:output:00model_8/lstm_3/while/maximum_iterations:output:0model_8/lstm_3/time:output:0'model_8/lstm_3/TensorArrayV2_1:handle:0model_8/lstm_3/zeros:output:0model_8/lstm_3/zeros_1:output:0'model_8/lstm_3/strided_slice_1:output:0Fmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:09model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resource;model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource:model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
model_8_lstm_3_while_body_60360*+
cond#R!
model_8_lstm_3_while_cond_60359*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
model_8/lstm_3/whileÓ
?model_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2A
?model_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape¥
1model_8/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStackmodel_8/lstm_3/while:output:3Hmodel_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype023
1model_8/lstm_3/TensorArrayV2Stack/TensorListStack
$model_8/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$model_8/lstm_3/strided_slice_3/stack
&model_8/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/lstm_3/strided_slice_3/stack_1
&model_8/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_3/stack_2õ
model_8/lstm_3/strided_slice_3StridedSlice:model_8/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-model_8/lstm_3/strided_slice_3/stack:output:0/model_8/lstm_3/strided_slice_3/stack_1:output:0/model_8/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2 
model_8/lstm_3/strided_slice_3
model_8/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_8/lstm_3/transpose_1/permâ
model_8/lstm_3/transpose_1	Transpose:model_8/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0(model_8/lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
model_8/lstm_3/transpose_1
model_8/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_3/runtime
model_8/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2!
model_8/dropout_5/dropout/ConstË
model_8/dropout_5/dropout/MulMul'model_8/lstm_3/strided_slice_3:output:0(model_8/dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
model_8/dropout_5/dropout/Mul
model_8/dropout_5/dropout/ShapeShape'model_8/lstm_3/strided_slice_3:output:0*
T0*
_output_shapes
:2!
model_8/dropout_5/dropout/Shapeë
6model_8/dropout_5/dropout/random_uniform/RandomUniformRandomUniform(model_8/dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype028
6model_8/dropout_5/dropout/random_uniform/RandomUniform
(model_8/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2*
(model_8/dropout_5/dropout/GreaterEqual/y
&model_8/dropout_5/dropout/GreaterEqualGreaterEqual?model_8/dropout_5/dropout/random_uniform/RandomUniform:output:01model_8/dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&model_8/dropout_5/dropout/GreaterEqual¶
model_8/dropout_5/dropout/CastCast*model_8/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
model_8/dropout_5/dropout/CastÃ
model_8/dropout_5/dropout/Mul_1Mul!model_8/dropout_5/dropout/Mul:z:0"model_8/dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
model_8/dropout_5/dropout/Mul_1¾
%model_8/dense_7/MatMul/ReadVariableOpReadVariableOp.model_8_dense_7_matmul_readvariableop_resource*
_output_shapes
:	ÈF*
dtype02'
%model_8/dense_7/MatMul/ReadVariableOpÀ
model_8/dense_7/MatMulMatMul#model_8/dropout_5/dropout/Mul_1:z:0-model_8/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/dense_7/MatMul¼
&model_8/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_8_dense_7_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02(
&model_8/dense_7/BiasAdd/ReadVariableOpÁ
model_8/dense_7/BiasAddBiasAdd model_8/dense_7/MatMul:product:0.model_8/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/dense_7/BiasAdd
model_8/dense_7/SigmoidSigmoid model_8/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/dense_7/Sigmoid
IdentityIdentitymodel_8/dense_7/Sigmoid:y:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp'^model_8/dense_7/BiasAdd/ReadVariableOp&^model_8/dense_7/MatMul/ReadVariableOp?^model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpG^model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpG^model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp2^model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp1^model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp3^model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^model_8/lstm_3/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2P
&model_8/dense_7/BiasAdd/ReadVariableOp&model_8/dense_7/BiasAdd/ReadVariableOp2N
%model_8/dense_7/MatMul/ReadVariableOp%model_8/dense_7/MatMul/ReadVariableOp2
>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp2
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpFmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp2
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpFmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp2f
1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2d
0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp2h
2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2,
model_8/lstm_3/whilemodel_8/lstm_3/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¡
Ö
'__inference_model_8_layer_call_fn_61363

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ý	
Ê
lstm_3_while_cond_61229*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_61229___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_61229___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_61229___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_61229___redundant_placeholder3
lstm_3_while_identity

lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
@
ô
while_body_62017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource¢(while/lstm_cell_3/BiasAdd/ReadVariableOp¢'while/lstm_cell_3/MatMul/ReadVariableOp¢)while/lstm_cell_3/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOpÔ
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMulÍ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp½
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMul_1´
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/addÅ
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpÁ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
while/lstm_cell_3/split
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_1
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul­
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_1¦
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/add_1
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_2ª
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
³
æ
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60038
input_11
dense_8_60014
dense_8_60016
model_8_60020
model_8_60022
model_8_60024
model_8_60026
model_8_60028
model_8_60030
model_8_60032
model_8_60034
identity¢dense_8/StatefulPartitionedCall¢model_8/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinput_11dense_8_60014dense_8_60016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_599142!
dense_8/StatefulPartitionedCallÿ
reshape_13/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_599432
reshape_13/PartitionedCall
model_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0model_8_60020model_8_60022model_8_60024model_8_60026model_8_60028model_8_60030model_8_60032model_8_60034*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598612!
model_8/StatefulPartitionedCallÀ
IdentityIdentity(model_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_11
¢
b
)__inference_dropout_5_layer_call_fn_62144

inputs
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_597012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¡
Ö
'__inference_model_8_layer_call_fn_61342

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ª
¾
while_cond_59575
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_59575___redundant_placeholder03
/while_while_cond_59575___redundant_placeholder13
/while_while_cond_59575___redundant_placeholder23
/while_while_cond_59575___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
ÒX
ê
A__inference_lstm_3_layer_call_and_return_conditional_losses_61631

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity¢"lstm_cell_3/BiasAdd/ReadVariableOp¢!lstm_cell_3/MatMul/ReadVariableOp¢#lstm_cell_3/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02#
!lstm_cell_3/MatMul/ReadVariableOpª
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul¹
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¦
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul_1
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimó
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_cell_3/split
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_1
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_1
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/add_1
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_2
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterí
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_61548*
condR
while_cond_61547*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeä
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
F:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
÷
a
E__inference_reshape_16_layer_call_and_return_conditional_losses_61477

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
F:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
ñ
÷
(__inference_T-GCN-WX_layer_call_fn_60091
input_11
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_600682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_11
î
ê
model_8_lstm_3_while_cond_60359:
6model_8_lstm_3_while_model_8_lstm_3_while_loop_counter@
<model_8_lstm_3_while_model_8_lstm_3_while_maximum_iterations$
 model_8_lstm_3_while_placeholder&
"model_8_lstm_3_while_placeholder_1&
"model_8_lstm_3_while_placeholder_2&
"model_8_lstm_3_while_placeholder_3<
8model_8_lstm_3_while_less_model_8_lstm_3_strided_slice_1Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60359___redundant_placeholder0Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60359___redundant_placeholder1Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60359___redundant_placeholder2Q
Mmodel_8_lstm_3_while_model_8_lstm_3_while_cond_60359___redundant_placeholder3!
model_8_lstm_3_while_identity
»
model_8/lstm_3/while/LessLess model_8_lstm_3_while_placeholder8model_8_lstm_3_while_less_model_8_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
model_8/lstm_3/while/Less
model_8/lstm_3/while/IdentityIdentitymodel_8/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
model_8/lstm_3/while/Identity"G
model_8_lstm_3_while_identity&model_8/lstm_3/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:

Û
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_62200

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2©
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity­

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1­

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1
Á
Ë
+__inference_lstm_cell_3_layer_call_fn_62248

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_586832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1
Ò·
ñ	
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60731

inputs-
)dense_8_tensordot_readvariableop_resource+
'dense_8_biasadd_readvariableop_resourceO
Kmodel_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resourceO
Kmodel_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resourceK
Gmodel_8_fixed_adjacency_graph_convolution_3_add_readvariableop_resource=
9model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resource?
;model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource>
:model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource2
.model_8_dense_7_matmul_readvariableop_resource3
/model_8_dense_7_biasadd_readvariableop_resource
identity¢dense_8/BiasAdd/ReadVariableOp¢ dense_8/Tensordot/ReadVariableOp¢&model_8/dense_7/BiasAdd/ReadVariableOp¢%model_8/dense_7/MatMul/ReadVariableOp¢>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp¢Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp¢Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp¢1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp¢0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp¢2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp¢model_8/lstm_3/while®
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axes
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_8/Tensordot/freeh
dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_8/Tensordot/Shape
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axisù
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axisÿ
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1¨
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axisØ
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat¬
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stack¬
dense_8/Tensordot/transpose	Transposeinputs!dense_8/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_8/Tensordot/transpose¿
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_8/Tensordot/Reshape¾
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/Tensordot/MatMul
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/Const_2
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axiså
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1´
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_8/Tensordot¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp«
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_8/BiasAddl
reshape_13/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_13/Shape
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2¤
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_13/Reshape/shape/1z
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/2×
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape¦
reshape_13/ReshapeReshapedense_8/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_13/Reshape
'model_8/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_8/tf.expand_dims_3/ExpandDims/dimá
#model_8/tf.expand_dims_3/ExpandDims
ExpandDimsreshape_13/Reshape:output:00model_8/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2%
#model_8/tf.expand_dims_3/ExpandDims
model_8/reshape_14/ShapeShape,model_8/tf.expand_dims_3/ExpandDims:output:0*
T0*
_output_shapes
:2
model_8/reshape_14/Shape
&model_8/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_14/strided_slice/stack
(model_8/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_14/strided_slice/stack_1
(model_8/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_14/strided_slice/stack_2Ô
 model_8/reshape_14/strided_sliceStridedSlice!model_8/reshape_14/Shape:output:0/model_8/reshape_14/strided_slice/stack:output:01model_8/reshape_14/strided_slice/stack_1:output:01model_8/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_14/strided_slice
"model_8/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_14/Reshape/shape/1
"model_8/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_8/reshape_14/Reshape/shape/2ÿ
 model_8/reshape_14/Reshape/shapePack)model_8/reshape_14/strided_slice:output:0+model_8/reshape_14/Reshape/shape/1:output:0+model_8/reshape_14/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_14/Reshape/shapeÒ
model_8/reshape_14/ReshapeReshape,model_8/tf.expand_dims_3/ExpandDims:output:0)model_8/reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/reshape_14/ReshapeÍ
:model_8/fixed_adjacency_graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2<
:model_8/fixed_adjacency_graph_convolution_3/transpose/perm
5model_8/fixed_adjacency_graph_convolution_3/transpose	Transpose#model_8/reshape_14/Reshape:output:0Cmodel_8/fixed_adjacency_graph_convolution_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF27
5model_8/fixed_adjacency_graph_convolution_3/transposeÏ
1model_8/fixed_adjacency_graph_convolution_3/ShapeShape9model_8/fixed_adjacency_graph_convolution_3/transpose:y:0*
T0*
_output_shapes
:23
1model_8/fixed_adjacency_graph_convolution_3/Shapeà
3model_8/fixed_adjacency_graph_convolution_3/unstackUnpack:model_8/fixed_adjacency_graph_convolution_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num25
3model_8/fixed_adjacency_graph_convolution_3/unstack
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02D
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOp»
3model_8/fixed_adjacency_graph_convolution_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   25
3model_8/fixed_adjacency_graph_convolution_3/Shape_1ä
5model_8/fixed_adjacency_graph_convolution_3/unstack_1Unpack<model_8/fixed_adjacency_graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num27
5model_8/fixed_adjacency_graph_convolution_3/unstack_1Ç
9model_8/fixed_adjacency_graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2;
9model_8/fixed_adjacency_graph_convolution_3/Reshape/shape¦
3model_8/fixed_adjacency_graph_convolution_3/ReshapeReshape9model_8/fixed_adjacency_graph_convolution_3/transpose:y:0Bmodel_8/fixed_adjacency_graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF25
3model_8/fixed_adjacency_graph_convolution_3/Reshape
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02H
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpÍ
<model_8/fixed_adjacency_graph_convolution_3/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_8/fixed_adjacency_graph_convolution_3/transpose_1/perm¿
7model_8/fixed_adjacency_graph_convolution_3/transpose_1	TransposeNmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp:value:0Emodel_8/fixed_adjacency_graph_convolution_3/transpose_1/perm:output:0*
T0*
_output_shapes

:FF29
7model_8/fixed_adjacency_graph_convolution_3/transpose_1Ë
;model_8/fixed_adjacency_graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_1/shape¥
5model_8/fixed_adjacency_graph_convolution_3/Reshape_1Reshape;model_8/fixed_adjacency_graph_convolution_3/transpose_1:y:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_1¢
2model_8/fixed_adjacency_graph_convolution_3/MatMulMatMul<model_8/fixed_adjacency_graph_convolution_3/Reshape:output:0>model_8/fixed_adjacency_graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF24
2model_8/fixed_adjacency_graph_convolution_3/MatMulÀ
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1À
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2þ
;model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shapePack<model_8/fixed_adjacency_graph_convolution_3/unstack:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/1:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape³
5model_8/fixed_adjacency_graph_convolution_3/Reshape_2Reshape<model_8/fixed_adjacency_graph_convolution_3/MatMul:product:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_2Ñ
<model_8/fixed_adjacency_graph_convolution_3/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<model_8/fixed_adjacency_graph_convolution_3/transpose_2/perm¼
7model_8/fixed_adjacency_graph_convolution_3/transpose_2	Transpose>model_8/fixed_adjacency_graph_convolution_3/Reshape_2:output:0Emodel_8/fixed_adjacency_graph_convolution_3/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF29
7model_8/fixed_adjacency_graph_convolution_3/transpose_2Õ
3model_8/fixed_adjacency_graph_convolution_3/Shape_2Shape;model_8/fixed_adjacency_graph_convolution_3/transpose_2:y:0*
T0*
_output_shapes
:25
3model_8/fixed_adjacency_graph_convolution_3/Shape_2æ
5model_8/fixed_adjacency_graph_convolution_3/unstack_2Unpack<model_8/fixed_adjacency_graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num27
5model_8/fixed_adjacency_graph_convolution_3/unstack_2
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02D
Bmodel_8/fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOp»
3model_8/fixed_adjacency_graph_convolution_3/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   25
3model_8/fixed_adjacency_graph_convolution_3/Shape_3ä
5model_8/fixed_adjacency_graph_convolution_3/unstack_3Unpack<model_8/fixed_adjacency_graph_convolution_3/Shape_3:output:0*
T0*
_output_shapes
: : *	
num27
5model_8/fixed_adjacency_graph_convolution_3/unstack_3Ë
;model_8/fixed_adjacency_graph_convolution_3/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_3/shape®
5model_8/fixed_adjacency_graph_convolution_3/Reshape_3Reshape;model_8/fixed_adjacency_graph_convolution_3/transpose_2:y:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_3
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02H
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpÍ
<model_8/fixed_adjacency_graph_convolution_3/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_8/fixed_adjacency_graph_convolution_3/transpose_3/perm¿
7model_8/fixed_adjacency_graph_convolution_3/transpose_3	TransposeNmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp:value:0Emodel_8/fixed_adjacency_graph_convolution_3/transpose_3/perm:output:0*
T0*
_output_shapes

:
29
7model_8/fixed_adjacency_graph_convolution_3/transpose_3Ë
;model_8/fixed_adjacency_graph_convolution_3/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_4/shape¥
5model_8/fixed_adjacency_graph_convolution_3/Reshape_4Reshape;model_8/fixed_adjacency_graph_convolution_3/transpose_3:y:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_4/shape:output:0*
T0*
_output_shapes

:
27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_4¨
4model_8/fixed_adjacency_graph_convolution_3/MatMul_1MatMul>model_8/fixed_adjacency_graph_convolution_3/Reshape_3:output:0>model_8/fixed_adjacency_graph_convolution_3/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
26
4model_8/fixed_adjacency_graph_convolution_3/MatMul_1À
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1À
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2?
=model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2
;model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shapePack>model_8/fixed_adjacency_graph_convolution_3/unstack_2:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/1:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_8/fixed_adjacency_graph_convolution_3/Reshape_5/shapeµ
5model_8/fixed_adjacency_graph_convolution_3/Reshape_5Reshape>model_8/fixed_adjacency_graph_convolution_3/MatMul_1:product:0Dmodel_8/fixed_adjacency_graph_convolution_3/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
27
5model_8/fixed_adjacency_graph_convolution_3/Reshape_5
>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpReadVariableOpGmodel_8_fixed_adjacency_graph_convolution_3_add_readvariableop_resource*
_output_shapes

:F*
dtype02@
>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp©
/model_8/fixed_adjacency_graph_convolution_3/addAddV2>model_8/fixed_adjacency_graph_convolution_3/Reshape_5:output:0Fmodel_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
21
/model_8/fixed_adjacency_graph_convolution_3/add
model_8/reshape_15/ShapeShape3model_8/fixed_adjacency_graph_convolution_3/add:z:0*
T0*
_output_shapes
:2
model_8/reshape_15/Shape
&model_8/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_15/strided_slice/stack
(model_8/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_15/strided_slice/stack_1
(model_8/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_15/strided_slice/stack_2Ô
 model_8/reshape_15/strided_sliceStridedSlice!model_8/reshape_15/Shape:output:0/model_8/reshape_15/strided_slice/stack:output:01model_8/reshape_15/strided_slice/stack_1:output:01model_8/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_15/strided_slice
"model_8/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_15/Reshape/shape/1
"model_8/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"model_8/reshape_15/Reshape/shape/2
"model_8/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_8/reshape_15/Reshape/shape/3¬
 model_8/reshape_15/Reshape/shapePack)model_8/reshape_15/strided_slice:output:0+model_8/reshape_15/Reshape/shape/1:output:0+model_8/reshape_15/Reshape/shape/2:output:0+model_8/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_15/Reshape/shapeÝ
model_8/reshape_15/ReshapeReshape3model_8/fixed_adjacency_graph_convolution_3/add:z:0)model_8/reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
model_8/reshape_15/Reshape
 model_8/permute_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 model_8/permute_3/transpose/permÑ
model_8/permute_3/transpose	Transpose#model_8/reshape_15/Reshape:output:0)model_8/permute_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
model_8/permute_3/transpose
model_8/reshape_16/ShapeShapemodel_8/permute_3/transpose:y:0*
T0*
_output_shapes
:2
model_8/reshape_16/Shape
&model_8/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_16/strided_slice/stack
(model_8/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_16/strided_slice/stack_1
(model_8/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_16/strided_slice/stack_2Ô
 model_8/reshape_16/strided_sliceStridedSlice!model_8/reshape_16/Shape:output:0/model_8/reshape_16/strided_slice/stack:output:01model_8/reshape_16/strided_slice/stack_1:output:01model_8/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_16/strided_slice
"model_8/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"model_8/reshape_16/Reshape/shape/1
"model_8/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_16/Reshape/shape/2ÿ
 model_8/reshape_16/Reshape/shapePack)model_8/reshape_16/strided_slice:output:0+model_8/reshape_16/Reshape/shape/1:output:0+model_8/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_16/Reshape/shapeÅ
model_8/reshape_16/ReshapeReshapemodel_8/permute_3/transpose:y:0)model_8/reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
model_8/reshape_16/Reshape
model_8/lstm_3/ShapeShape#model_8/reshape_16/Reshape:output:0*
T0*
_output_shapes
:2
model_8/lstm_3/Shape
"model_8/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_8/lstm_3/strided_slice/stack
$model_8/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_8/lstm_3/strided_slice/stack_1
$model_8/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_8/lstm_3/strided_slice/stack_2¼
model_8/lstm_3/strided_sliceStridedSlicemodel_8/lstm_3/Shape:output:0+model_8/lstm_3/strided_slice/stack:output:0-model_8/lstm_3/strided_slice/stack_1:output:0-model_8/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_8/lstm_3/strided_slice{
model_8/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
model_8/lstm_3/zeros/mul/y¨
model_8/lstm_3/zeros/mulMul%model_8/lstm_3/strided_slice:output:0#model_8/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros/mul}
model_8/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
model_8/lstm_3/zeros/Less/y£
model_8/lstm_3/zeros/LessLessmodel_8/lstm_3/zeros/mul:z:0$model_8/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros/Less
model_8/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
model_8/lstm_3/zeros/packed/1¿
model_8/lstm_3/zeros/packedPack%model_8/lstm_3/strided_slice:output:0&model_8/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_8/lstm_3/zeros/packed}
model_8/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_3/zeros/Const²
model_8/lstm_3/zerosFill$model_8/lstm_3/zeros/packed:output:0#model_8/lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
model_8/lstm_3/zeros
model_8/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
model_8/lstm_3/zeros_1/mul/y®
model_8/lstm_3/zeros_1/mulMul%model_8/lstm_3/strided_slice:output:0%model_8/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros_1/mul
model_8/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
model_8/lstm_3/zeros_1/Less/y«
model_8/lstm_3/zeros_1/LessLessmodel_8/lstm_3/zeros_1/mul:z:0&model_8/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_3/zeros_1/Less
model_8/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2!
model_8/lstm_3/zeros_1/packed/1Å
model_8/lstm_3/zeros_1/packedPack%model_8/lstm_3/strided_slice:output:0(model_8/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_8/lstm_3/zeros_1/packed
model_8/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_3/zeros_1/Constº
model_8/lstm_3/zeros_1Fill&model_8/lstm_3/zeros_1/packed:output:0%model_8/lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
model_8/lstm_3/zeros_1
model_8/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_8/lstm_3/transpose/permÄ
model_8/lstm_3/transpose	Transpose#model_8/reshape_16/Reshape:output:0&model_8/lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
model_8/lstm_3/transpose|
model_8/lstm_3/Shape_1Shapemodel_8/lstm_3/transpose:y:0*
T0*
_output_shapes
:2
model_8/lstm_3/Shape_1
$model_8/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_8/lstm_3/strided_slice_1/stack
&model_8/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_1/stack_1
&model_8/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_1/stack_2È
model_8/lstm_3/strided_slice_1StridedSlicemodel_8/lstm_3/Shape_1:output:0-model_8/lstm_3/strided_slice_1/stack:output:0/model_8/lstm_3/strided_slice_1/stack_1:output:0/model_8/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
model_8/lstm_3/strided_slice_1£
*model_8/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*model_8/lstm_3/TensorArrayV2/element_shapeî
model_8/lstm_3/TensorArrayV2TensorListReserve3model_8/lstm_3/TensorArrayV2/element_shape:output:0'model_8/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_8/lstm_3/TensorArrayV2Ý
Dmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2F
Dmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape´
6model_8/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_8/lstm_3/transpose:y:0Mmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6model_8/lstm_3/TensorArrayUnstack/TensorListFromTensor
$model_8/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_8/lstm_3/strided_slice_2/stack
&model_8/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_2/stack_1
&model_8/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_2/stack_2Ö
model_8/lstm_3/strided_slice_2StridedSlicemodel_8/lstm_3/transpose:y:0-model_8/lstm_3/strided_slice_2/stack:output:0/model_8/lstm_3/strided_slice_2/stack_1:output:0/model_8/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2 
model_8/lstm_3/strided_slice_2ß
0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype022
0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOpæ
!model_8/lstm_3/lstm_cell_3/MatMulMatMul'model_8/lstm_3/strided_slice_2:output:08model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!model_8/lstm_3/lstm_cell_3/MatMulæ
2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype024
2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpâ
#model_8/lstm_3/lstm_cell_3/MatMul_1MatMulmodel_8/lstm_3/zeros:output:0:model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#model_8/lstm_3/lstm_cell_3/MatMul_1Ø
model_8/lstm_3/lstm_cell_3/addAddV2+model_8/lstm_3/lstm_cell_3/MatMul:product:0-model_8/lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
model_8/lstm_3/lstm_cell_3/addÞ
1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype023
1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpå
"model_8/lstm_3/lstm_cell_3/BiasAddBiasAdd"model_8/lstm_3/lstm_cell_3/add:z:09model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"model_8/lstm_3/lstm_cell_3/BiasAdd
 model_8/lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_8/lstm_3/lstm_cell_3/Const
*model_8/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_8/lstm_3/lstm_cell_3/split/split_dim¯
 model_8/lstm_3/lstm_cell_3/splitSplit3model_8/lstm_3/lstm_cell_3/split/split_dim:output:0+model_8/lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2"
 model_8/lstm_3/lstm_cell_3/split±
"model_8/lstm_3/lstm_cell_3/SigmoidSigmoid)model_8/lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"model_8/lstm_3/lstm_cell_3/Sigmoidµ
$model_8/lstm_3/lstm_cell_3/Sigmoid_1Sigmoid)model_8/lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$model_8/lstm_3/lstm_cell_3/Sigmoid_1Å
model_8/lstm_3/lstm_cell_3/mulMul(model_8/lstm_3/lstm_cell_3/Sigmoid_1:y:0model_8/lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
model_8/lstm_3/lstm_cell_3/mulÑ
 model_8/lstm_3/lstm_cell_3/mul_1Mul&model_8/lstm_3/lstm_cell_3/Sigmoid:y:0)model_8/lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 model_8/lstm_3/lstm_cell_3/mul_1Ê
 model_8/lstm_3/lstm_cell_3/add_1AddV2"model_8/lstm_3/lstm_cell_3/mul:z:0$model_8/lstm_3/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 model_8/lstm_3/lstm_cell_3/add_1µ
$model_8/lstm_3/lstm_cell_3/Sigmoid_2Sigmoid)model_8/lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$model_8/lstm_3/lstm_cell_3/Sigmoid_2Î
 model_8/lstm_3/lstm_cell_3/mul_2Mul(model_8/lstm_3/lstm_cell_3/Sigmoid_2:y:0$model_8/lstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 model_8/lstm_3/lstm_cell_3/mul_2­
,model_8/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2.
,model_8/lstm_3/TensorArrayV2_1/element_shapeô
model_8/lstm_3/TensorArrayV2_1TensorListReserve5model_8/lstm_3/TensorArrayV2_1/element_shape:output:0'model_8/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_8/lstm_3/TensorArrayV2_1l
model_8/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_8/lstm_3/time
'model_8/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_8/lstm_3/while/maximum_iterations
!model_8/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model_8/lstm_3/while/loop_counterÎ
model_8/lstm_3/whileWhile*model_8/lstm_3/while/loop_counter:output:00model_8/lstm_3/while/maximum_iterations:output:0model_8/lstm_3/time:output:0'model_8/lstm_3/TensorArrayV2_1:handle:0model_8/lstm_3/zeros:output:0model_8/lstm_3/zeros_1:output:0'model_8/lstm_3/strided_slice_1:output:0Fmodel_8/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:09model_8_lstm_3_lstm_cell_3_matmul_readvariableop_resource;model_8_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource:model_8_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
model_8_lstm_3_while_body_60640*+
cond#R!
model_8_lstm_3_while_cond_60639*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
model_8/lstm_3/whileÓ
?model_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2A
?model_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape¥
1model_8/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStackmodel_8/lstm_3/while:output:3Hmodel_8/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype023
1model_8/lstm_3/TensorArrayV2Stack/TensorListStack
$model_8/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$model_8/lstm_3/strided_slice_3/stack
&model_8/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/lstm_3/strided_slice_3/stack_1
&model_8/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_3/strided_slice_3/stack_2õ
model_8/lstm_3/strided_slice_3StridedSlice:model_8/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-model_8/lstm_3/strided_slice_3/stack:output:0/model_8/lstm_3/strided_slice_3/stack_1:output:0/model_8/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2 
model_8/lstm_3/strided_slice_3
model_8/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_8/lstm_3/transpose_1/permâ
model_8/lstm_3/transpose_1	Transpose:model_8/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0(model_8/lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
model_8/lstm_3/transpose_1
model_8/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_3/runtime 
model_8/dropout_5/IdentityIdentity'model_8/lstm_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
model_8/dropout_5/Identity¾
%model_8/dense_7/MatMul/ReadVariableOpReadVariableOp.model_8_dense_7_matmul_readvariableop_resource*
_output_shapes
:	ÈF*
dtype02'
%model_8/dense_7/MatMul/ReadVariableOpÀ
model_8/dense_7/MatMulMatMul#model_8/dropout_5/Identity:output:0-model_8/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/dense_7/MatMul¼
&model_8/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_8_dense_7_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02(
&model_8/dense_7/BiasAdd/ReadVariableOpÁ
model_8/dense_7/BiasAddBiasAdd model_8/dense_7/MatMul:product:0.model_8/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/dense_7/BiasAdd
model_8/dense_7/SigmoidSigmoid model_8/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_8/dense_7/Sigmoid
IdentityIdentitymodel_8/dense_7/Sigmoid:y:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp'^model_8/dense_7/BiasAdd/ReadVariableOp&^model_8/dense_7/MatMul/ReadVariableOp?^model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOpG^model_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpG^model_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp2^model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp1^model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp3^model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^model_8/lstm_3/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2P
&model_8/dense_7/BiasAdd/ReadVariableOp&model_8/dense_7/BiasAdd/ReadVariableOp2N
%model_8/dense_7/MatMul/ReadVariableOp%model_8/dense_7/MatMul/ReadVariableOp2
>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp>model_8/fixed_adjacency_graph_convolution_3/add/ReadVariableOp2
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpFmodel_8/fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp2
Fmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpFmodel_8/fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp2f
1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp1model_8/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2d
0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp0model_8/lstm_3/lstm_cell_3/MatMul/ReadVariableOp2h
2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2model_8/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2,
model_8/lstm_3/whilemodel_8/lstm_3/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
÷
a
E__inference_reshape_16_layer_call_and_return_conditional_losses_59354

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
F:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
¸&
Ë
B__inference_model_8_layer_call_and_return_conditional_losses_59861

inputs-
)fixed_adjacency_graph_convolution_3_59837-
)fixed_adjacency_graph_convolution_3_59839-
)fixed_adjacency_graph_convolution_3_59841
lstm_3_59847
lstm_3_59849
lstm_3_59851
dense_7_59855
dense_7_59857
identity¢dense_7/StatefulPartitionedCall¢;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim´
tf.expand_dims_3/ExpandDims
ExpandDimsinputs(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_3/ExpandDimsû
reshape_14/PartitionedCallPartitionedCall$tf.expand_dims_3/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_14_layer_call_and_return_conditional_losses_592372
reshape_14/PartitionedCallæ
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0)fixed_adjacency_graph_convolution_3_59837)fixed_adjacency_graph_convolution_3_59839)fixed_adjacency_graph_convolution_3_59841*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *g
fbR`
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_592982=
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall
reshape_15/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_15_layer_call_and_return_conditional_losses_593322
reshape_15/PartitionedCallû
permute_3/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_permute_3_layer_call_and_return_conditional_losses_586062
permute_3/PartitionedCallù
reshape_16/PartitionedCallPartitionedCall"permute_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_593542
reshape_16/PartitionedCallµ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0lstm_3_59847lstm_3_59849lstm_3_59851*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_596592 
lstm_3/StatefulPartitionedCallø
dropout_5/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_597062
dropout_5/PartitionedCall¨
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_59855dense_7_59857*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_597302!
dense_7/StatefulPartitionedCallý
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_3/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall;fixed_adjacency_graph_convolution_3/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ª
¾
while_cond_61547
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_61547___redundant_placeholder03
/while_while_cond_61547___redundant_placeholder13
/while_while_cond_61547___redundant_placeholder23
/while_while_cond_61547___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
÷
a
E__inference_reshape_15_layer_call_and_return_conditional_losses_61459

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1m
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿF
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF

 
_user_specified_nameinputs
ë
á
B__inference_dense_8_layer_call_and_return_conditional_losses_59914

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
BiasAdd 
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
³
æ
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60011
input_11
dense_8_59925
dense_8_59927
model_8_59993
model_8_59995
model_8_59997
model_8_59999
model_8_60001
model_8_60003
model_8_60005
model_8_60007
identity¢dense_8/StatefulPartitionedCall¢model_8/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinput_11dense_8_59925dense_8_59927*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_599142!
dense_8/StatefulPartitionedCallÿ
reshape_13/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_13_layer_call_and_return_conditional_losses_599432
reshape_13/PartitionedCall
model_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_13/PartitionedCall:output:0model_8_59993model_8_59995model_8_59997model_8_59999model_8_60001model_8_60003model_8_60005model_8_60007*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_598102!
model_8/StatefulPartitionedCallÀ
IdentityIdentity(model_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^model_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿF::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_11
î
a
E__inference_reshape_13_layer_call_and_return_conditional_losses_60833

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ë
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_59706

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
î
a
E__inference_reshape_14_layer_call_and_return_conditional_losses_61376

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¬
F
*__inference_reshape_15_layer_call_fn_61464

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_15_layer_call_and_return_conditional_losses_593322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿF
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF

 
_user_specified_nameinputs

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_59701

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ÒX
ê
A__inference_lstm_3_layer_call_and_return_conditional_losses_59510

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity¢"lstm_cell_3/BiasAdd/ReadVariableOp¢!lstm_cell_3/MatMul/ReadVariableOp¢#lstm_cell_3/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02#
!lstm_cell_3/MatMul/ReadVariableOpª
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul¹
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¦
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/MatMul_1
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimó
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_cell_3/split
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_1
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_1
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/add_1
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/Sigmoid_2
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_3/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterí
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_59427*
condR
while_cond_59426*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeä
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
F:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
ý	
Ê
lstm_3_while_cond_60984*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_60984___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_60984___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_60984___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_60984___redundant_placeholder3
lstm_3_while_identity

lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
@
ô
while_body_59576
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource¢(while/lstm_cell_3/BiasAdd/ReadVariableOp¢'while/lstm_cell_3/MatMul/ReadVariableOp¢)while/lstm_cell_3/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOpÔ
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMulÍ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp½
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMul_1´
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/addÅ
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpÁ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
while/lstm_cell_3/split
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_1
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul­
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_1¦
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/add_1
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_2ª
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
¾,
¹
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_61434
features#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource
add_readvariableop_resource
identity¢add/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposefeaturestranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:FF2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:
2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:
2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
	Reshape_5
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
add®
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2(
add/ReadVariableOpadd/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
features
íé
È
B__inference_model_8_layer_call_and_return_conditional_losses_61321

inputsG
Cfixed_adjacency_graph_convolution_3_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_3_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_3_add_readvariableop_resource5
1lstm_3_lstm_cell_3_matmul_readvariableop_resource7
3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource6
2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢6fixed_adjacency_graph_convolution_3/add/ReadVariableOp¢>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp¢>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp¢)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp¢(lstm_3/lstm_cell_3/MatMul/ReadVariableOp¢*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp¢lstm_3/while
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim´
tf.expand_dims_3/ExpandDims
ExpandDimsinputs(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_3/ExpandDimsx
reshape_14/ShapeShape$tf.expand_dims_3/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_14/Shape
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stack
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2¤
reshape_14/strided_sliceStridedSlicereshape_14/Shape:output:0'reshape_14/strided_slice/stack:output:0)reshape_14/strided_slice/stack_1:output:0)reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_14/strided_slicez
reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_14/Reshape/shape/1z
reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/2×
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shape²
reshape_14/ReshapeReshape$tf.expand_dims_3/ExpandDims:output:0!reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_14/Reshape½
2fixed_adjacency_graph_convolution_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_3/transpose/permû
-fixed_adjacency_graph_convolution_3/transpose	Transposereshape_14/Reshape:output:0;fixed_adjacency_graph_convolution_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2/
-fixed_adjacency_graph_convolution_3/transpose·
)fixed_adjacency_graph_convolution_3/ShapeShape1fixed_adjacency_graph_convolution_3/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_3/ShapeÈ
+fixed_adjacency_graph_convolution_3/unstackUnpack2fixed_adjacency_graph_convolution_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_3/unstackü
:fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_3/Shape_1/ReadVariableOp«
+fixed_adjacency_graph_convolution_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_3/Shape_1Ì
-fixed_adjacency_graph_convolution_3/unstack_1Unpack4fixed_adjacency_graph_convolution_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_3/unstack_1·
1fixed_adjacency_graph_convolution_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   23
1fixed_adjacency_graph_convolution_3/Reshape/shape
+fixed_adjacency_graph_convolution_3/ReshapeReshape1fixed_adjacency_graph_convolution_3/transpose:y:0:fixed_adjacency_graph_convolution_3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2-
+fixed_adjacency_graph_convolution_3/Reshape
>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp½
4fixed_adjacency_graph_convolution_3/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_3/transpose_1/perm
/fixed_adjacency_graph_convolution_3/transpose_1	TransposeFfixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_3/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_3/transpose_1»
3fixed_adjacency_graph_convolution_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ25
3fixed_adjacency_graph_convolution_3/Reshape_1/shape
-fixed_adjacency_graph_convolution_3/Reshape_1Reshape3fixed_adjacency_graph_convolution_3/transpose_1:y:0<fixed_adjacency_graph_convolution_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_3/Reshape_1
*fixed_adjacency_graph_convolution_3/MatMulMatMul4fixed_adjacency_graph_convolution_3/Reshape:output:06fixed_adjacency_graph_convolution_3/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2,
*fixed_adjacency_graph_convolution_3/MatMul°
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/1°
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_3/Reshape_2/shape/2Ö
3fixed_adjacency_graph_convolution_3/Reshape_2/shapePack4fixed_adjacency_graph_convolution_3/unstack:output:0>fixed_adjacency_graph_convolution_3/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_3/Reshape_2/shape
-fixed_adjacency_graph_convolution_3/Reshape_2Reshape4fixed_adjacency_graph_convolution_3/MatMul:product:0<fixed_adjacency_graph_convolution_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2/
-fixed_adjacency_graph_convolution_3/Reshape_2Á
4fixed_adjacency_graph_convolution_3/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_3/transpose_2/perm
/fixed_adjacency_graph_convolution_3/transpose_2	Transpose6fixed_adjacency_graph_convolution_3/Reshape_2:output:0=fixed_adjacency_graph_convolution_3/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF21
/fixed_adjacency_graph_convolution_3/transpose_2½
+fixed_adjacency_graph_convolution_3/Shape_2Shape3fixed_adjacency_graph_convolution_3/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_3/Shape_2Î
-fixed_adjacency_graph_convolution_3/unstack_2Unpack4fixed_adjacency_graph_convolution_3/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_3/unstack_2ü
:fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02<
:fixed_adjacency_graph_convolution_3/Shape_3/ReadVariableOp«
+fixed_adjacency_graph_convolution_3/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2-
+fixed_adjacency_graph_convolution_3/Shape_3Ì
-fixed_adjacency_graph_convolution_3/unstack_3Unpack4fixed_adjacency_graph_convolution_3/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_3/unstack_3»
3fixed_adjacency_graph_convolution_3/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   25
3fixed_adjacency_graph_convolution_3/Reshape_3/shape
-fixed_adjacency_graph_convolution_3/Reshape_3Reshape3fixed_adjacency_graph_convolution_3/transpose_2:y:0<fixed_adjacency_graph_convolution_3/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-fixed_adjacency_graph_convolution_3/Reshape_3
>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_3_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02@
>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp½
4fixed_adjacency_graph_convolution_3/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_3/transpose_3/perm
/fixed_adjacency_graph_convolution_3/transpose_3	TransposeFfixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_3/transpose_3/perm:output:0*
T0*
_output_shapes

:
21
/fixed_adjacency_graph_convolution_3/transpose_3»
3fixed_adjacency_graph_convolution_3/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ25
3fixed_adjacency_graph_convolution_3/Reshape_4/shape
-fixed_adjacency_graph_convolution_3/Reshape_4Reshape3fixed_adjacency_graph_convolution_3/transpose_3:y:0<fixed_adjacency_graph_convolution_3/Reshape_4/shape:output:0*
T0*
_output_shapes

:
2/
-fixed_adjacency_graph_convolution_3/Reshape_4
,fixed_adjacency_graph_convolution_3/MatMul_1MatMul6fixed_adjacency_graph_convolution_3/Reshape_3:output:06fixed_adjacency_graph_convolution_3/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2.
,fixed_adjacency_graph_convolution_3/MatMul_1°
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/1°
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
27
5fixed_adjacency_graph_convolution_3/Reshape_5/shape/2Ø
3fixed_adjacency_graph_convolution_3/Reshape_5/shapePack6fixed_adjacency_graph_convolution_3/unstack_2:output:0>fixed_adjacency_graph_convolution_3/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_3/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_3/Reshape_5/shape
-fixed_adjacency_graph_convolution_3/Reshape_5Reshape6fixed_adjacency_graph_convolution_3/MatMul_1:product:0<fixed_adjacency_graph_convolution_3/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2/
-fixed_adjacency_graph_convolution_3/Reshape_5ð
6fixed_adjacency_graph_convolution_3/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_3_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_3/add/ReadVariableOp
'fixed_adjacency_graph_convolution_3/addAddV26fixed_adjacency_graph_convolution_3/Reshape_5:output:0>fixed_adjacency_graph_convolution_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2)
'fixed_adjacency_graph_convolution_3/add
reshape_15/ShapeShape+fixed_adjacency_graph_convolution_3/add:z:0*
T0*
_output_shapes
:2
reshape_15/Shape
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2¤
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_15/Reshape/shape/1
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_15/Reshape/shape/2z
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/3ü
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape½
reshape_15/ReshapeReshape+fixed_adjacency_graph_convolution_3/add:z:0!reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
2
reshape_15/Reshape
permute_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_3/transpose/perm±
permute_3/transpose	Transposereshape_15/Reshape:output:0!permute_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
permute_3/transposek
reshape_16/ShapeShapepermute_3/transpose:y:0*
T0*
_output_shapes
:2
reshape_16/Shape
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2¤
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slice
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_16/Reshape/shape/2×
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape¥
reshape_16/ReshapeReshapepermute_3/transpose:y:0!reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F2
reshape_16/Reshapeg
lstm_3/ShapeShapereshape_16/Reshape:output:0*
T0*
_output_shapes
:2
lstm_3/Shape
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicek
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros/mul/y
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_3/zeros/Less/y
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessq
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros/packed/1
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/zeroso
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros_1/mul/y
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_3/zeros_1/Less/y
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lessu
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm_3/zeros_1/packed/1¥
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/zeros_1
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm¤
lstm_3/transpose	Transposereshape_16/Reshape:output:0lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿF2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stack
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_3/TensorArrayV2/element_shapeÎ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2Í
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensor
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stack
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2¦
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
lstm_3/strided_slice_2Ç
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	F *
dtype02*
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpÆ
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/MatMulÎ
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
È *
dtype02,
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpÂ
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/MatMul_1¸
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/MatMul:product:0%lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/addÆ
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02+
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpÅ
lstm_3/lstm_cell_3/BiasAddBiasAddlstm_3/lstm_cell_3/add:z:01lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_3/lstm_cell_3/BiasAddv
lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/lstm_cell_3/Const
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dim
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0#lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
lstm_3/lstm_cell_3/split
lstm_3/lstm_cell_3/SigmoidSigmoid!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/Sigmoid
lstm_3/lstm_cell_3/Sigmoid_1Sigmoid!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/Sigmoid_1¥
lstm_3/lstm_cell_3/mulMul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/mul±
lstm_3/lstm_cell_3/mul_1Mullstm_3/lstm_cell_3/Sigmoid:y:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/mul_1ª
lstm_3/lstm_cell_3/add_1AddV2lstm_3/lstm_cell_3/mul:z:0lstm_3/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/add_1
lstm_3/lstm_cell_3/Sigmoid_2Sigmoid!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/Sigmoid_2®
lstm_3/lstm_cell_3/mul_2Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0lstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_3/lstm_cell_3/mul_2
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2&
$lstm_3/TensorArrayV2_1/element_shapeÔ
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/time
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counterÖ
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_3_matmul_readvariableop_resource3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_3_while_body_61230*#
condR
lstm_3_while_cond_61229*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
lstm_3/whileÃ
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStack
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_3/strided_slice_3/stack
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2Å
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
lstm_3/strided_slice_3
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/permÂ
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtime
dropout_5/IdentityIdentitylstm_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_5/Identity¦
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	ÈF*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMuldropout_5/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_7/BiasAddy
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_7/Sigmoidö
IdentityIdentitydense_7/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_3/add/ReadVariableOp?^fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp*^lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_3/MatMul/ReadVariableOp+^lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^lstm_3/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_3/add/ReadVariableOp6fixed_adjacency_graph_convolution_3/add/ReadVariableOp2
>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_3/transpose_1/ReadVariableOp2
>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_3/transpose_3/ReadVariableOp2V
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp(lstm_3/lstm_cell_3/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs


&__inference_lstm_3_layer_call_fn_61802

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_596592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
F
 
_user_specified_nameinputs
ª
¾
while_cond_59139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_59139___redundant_placeholder03
/while_while_cond_59139___redundant_placeholder13
/while_while_cond_59139___redundant_placeholder23
/while_while_cond_59139___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
@
ô
while_body_61868
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource¢(while/lstm_cell_3/BiasAdd/ReadVariableOp¢'while/lstm_cell_3/MatMul/ReadVariableOp¢)while/lstm_cell_3/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	F *
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOpÔ
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMulÍ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp½
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/MatMul_1´
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/addÅ
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpÁ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split2
while/lstm_cell_3/split
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_1
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul­
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_1¦
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/add_1
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/Sigmoid_2ª
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_3/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: "±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
E
input_119
serving_default_input_11:0ÿÿÿÿÿÿÿÿÿF;
model_80
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿFtensorflow/serving/predict:Â¸
èT
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
¶_default_save_signature
·__call__
+¸&call_and_return_all_conditional_losses"²R
_tf_keras_networkR{"class_name": "Functional", "name": "T-GCN-WX", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "T-GCN-WX", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_13", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_13", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["input_12", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_14", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_14", "inbound_nodes": [[["tf.expand_dims_3", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_3", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_3", "inbound_nodes": [[["reshape_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_15", "inbound_nodes": [[["fixed_adjacency_graph_convolution_3", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_3", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_3", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_16", "inbound_nodes": [[["permute_3", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["reshape_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "name": "model_8", "inbound_nodes": [[["reshape_13", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["model_8", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "T-GCN-WX", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_13", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_13", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["input_12", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_14", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_14", "inbound_nodes": [[["tf.expand_dims_3", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_3", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_3", "inbound_nodes": [[["reshape_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_15", "inbound_nodes": [[["fixed_adjacency_graph_convolution_3", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_3", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_3", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_16", "inbound_nodes": [[["permute_3", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["reshape_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "name": "model_8", "inbound_nodes": [[["reshape_13", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["model_8", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
û"ø
_tf_keras_input_layerØ{"class_name": "InputLayer", "name": "input_11", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}}
ù

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12, 5]}}
ú
trainable_variables
regularization_losses
	variables
	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "Reshape", "name": "reshape_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_13", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
·?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
layer-8
layer_with_weights-2
layer-9
trainable_variables
 regularization_losses
!	variables
"	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"Ö<
_tf_keras_networkº<{"class_name": "Functional", "name": "model_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["input_12", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_14", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_14", "inbound_nodes": [[["tf.expand_dims_3", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_3", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_3", "inbound_nodes": [[["reshape_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_15", "inbound_nodes": [[["fixed_adjacency_graph_convolution_3", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_3", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_3", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_16", "inbound_nodes": [[["permute_3", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["reshape_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["input_12", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_14", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_14", "inbound_nodes": [[["tf.expand_dims_3", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_3", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_3", "inbound_nodes": [[["reshape_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_15", "inbound_nodes": [[["fixed_adjacency_graph_convolution_3", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_3", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_3", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_16", "inbound_nodes": [[["permute_3", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["reshape_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_7", 0, 0]]}}}

#iter

$beta_1

%beta_2
	&decay
'learning_ratem¤m¥(m¦)m§*m¨+m©,mª-m«.m¬v­v®(v¯)v°*v±+v²,v³-v´.vµ"
	optimizer
_
0
1
(2
)3
*4
+5
,6
-7
.8"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
(2
)3
/4
*5
+6
,7
-8
.9"
trackable_list_wrapper
Î
0metrics

1layers
trainable_variables
regularization_losses
2layer_regularization_losses
3layer_metrics
4non_trainable_variables
	variables
·__call__
¶_default_save_signature
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
-
¿serving_default"
signature_map
 :2dense_8/kernel
:2dense_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

5layers
6metrics
trainable_variables
regularization_losses
7layer_regularization_losses
8layer_metrics
9non_trainable_variables
	variables
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

:layers
;metrics
trainable_variables
regularization_losses
<layer_regularization_losses
=layer_metrics
>non_trainable_variables
	variables
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
õ"ò
_tf_keras_input_layerÒ{"class_name": "InputLayer", "name": "input_12", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}}
ä
?	keras_api"Ò
_tf_keras_layer¸{"class_name": "TFOpLambda", "name": "tf.expand_dims_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
ú
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "Reshape", "name": "reshape_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_14", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
Ä
/A

(kernel
)bias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layerü{"class_name": "FixedAdjacencyGraphConvolution", "name": "fixed_adjacency_graph_convolution_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_adjacency_graph_convolution_3", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}}
ý
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Reshape", "name": "reshape_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}}

Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "Permute", "name": "permute_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "permute_3", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ú
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
È__call__
+É&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "Reshape", "name": "reshape_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}}
Â
Tcell
U
state_spec
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"

_tf_keras_rnn_layerù	{"class_name": "LSTM", "name": "lstm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 70]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 70]}}
ç
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
÷

-kernel
.bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
Q
(0
)1
*2
+3
,4
-5
.6"
trackable_list_wrapper
 "
trackable_list_wrapper
X
(0
)1
/2
*3
+4
,5
-6
.7"
trackable_list_wrapper
°
bmetrics

clayers
trainable_variables
 regularization_losses
dlayer_regularization_losses
elayer_metrics
fnon_trainable_variables
!	variables
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<::
2*fixed_adjacency_graph_convolution_3/kernel
::8F2(fixed_adjacency_graph_convolution_3/bias
,:*	F 2lstm_3/lstm_cell_3/kernel
7:5
È 2#lstm_3/lstm_cell_3/recurrent_kernel
&:$ 2lstm_3/lstm_cell_3/bias
!:	ÈF2dense_7/kernel
:F2dense_7/bias
5:3FF2%fixed_adjacency_graph_convolution_3/A
.
g0
h1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

ilayers
jmetrics
@trainable_variables
Aregularization_losses
klayer_regularization_losses
llayer_metrics
mnon_trainable_variables
B	variables
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
/2"
trackable_list_wrapper
°

nlayers
ometrics
Dtrainable_variables
Eregularization_losses
player_regularization_losses
qlayer_metrics
rnon_trainable_variables
F	variables
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

slayers
tmetrics
Htrainable_variables
Iregularization_losses
ulayer_regularization_losses
vlayer_metrics
wnon_trainable_variables
J	variables
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

xlayers
ymetrics
Ltrainable_variables
Mregularization_losses
zlayer_regularization_losses
{layer_metrics
|non_trainable_variables
N	variables
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²

}layers
~metrics
Ptrainable_variables
Qregularization_losses
layer_regularization_losses
layer_metrics
non_trainable_variables
R	variables
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
±

*kernel
+recurrent_kernel
,bias
trainable_variables
regularization_losses
	variables
	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "LSTMCell", "name": "lstm_cell_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_3", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
Â
metrics
layers
Vtrainable_variables
Wregularization_losses
 layer_regularization_losses
layer_metrics
non_trainable_variables
states
X	variables
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
metrics
Ztrainable_variables
[regularization_losses
 layer_regularization_losses
layer_metrics
non_trainable_variables
\	variables
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
µ
layers
metrics
^trainable_variables
_regularization_losses
 layer_regularization_losses
layer_metrics
non_trainable_variables
`	variables
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
/0"
trackable_list_wrapper
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ø

total

count

_fn_kwargs
	variables
	keras_api"¬
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
¸
layers
 metrics
trainable_variables
regularization_losses
 ¡layer_regularization_losses
¢layer_metrics
£non_trainable_variables
	variables
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
%:#2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
A:?
21Adam/fixed_adjacency_graph_convolution_3/kernel/m
?:=F2/Adam/fixed_adjacency_graph_convolution_3/bias/m
1:/	F 2 Adam/lstm_3/lstm_cell_3/kernel/m
<::
È 2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
+:) 2Adam/lstm_3/lstm_cell_3/bias/m
&:$	ÈF2Adam/dense_7/kernel/m
:F2Adam/dense_7/bias/m
%:#2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
A:?
21Adam/fixed_adjacency_graph_convolution_3/kernel/v
?:=F2/Adam/fixed_adjacency_graph_convolution_3/bias/v
1:/	F 2 Adam/lstm_3/lstm_cell_3/kernel/v
<::
È 2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
+:) 2Adam/lstm_3/lstm_cell_3/bias/v
&:$	ÈF2Adam/dense_7/kernel/v
:F2Adam/dense_7/bias/v
ç2ä
 __inference__wrapped_model_58599¿
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª */¢,
*'
input_11ÿÿÿÿÿÿÿÿÿF
î2ë
(__inference_T-GCN-WX_layer_call_fn_60756
(__inference_T-GCN-WX_layer_call_fn_60781
(__inference_T-GCN-WX_layer_call_fn_60091
(__inference_T-GCN-WX_layer_call_fn_60143À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60731
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60458
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60011
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60038À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_dense_8_layer_call_fn_60820¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_8_layer_call_and_return_conditional_losses_60811¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_reshape_13_layer_call_fn_60838¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_13_layer_call_and_return_conditional_losses_60833¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
'__inference_model_8_layer_call_fn_59829
'__inference_model_8_layer_call_fn_59880
'__inference_model_8_layer_call_fn_61363
'__inference_model_8_layer_call_fn_61342À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_8_layer_call_and_return_conditional_losses_61321
B__inference_model_8_layer_call_and_return_conditional_losses_61083
B__inference_model_8_layer_call_and_return_conditional_losses_59747
B__inference_model_8_layer_call_and_return_conditional_losses_59777À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
#__inference_signature_wrapper_60178input_11"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_reshape_14_layer_call_fn_61381¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_14_layer_call_and_return_conditional_losses_61376¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
C__inference_fixed_adjacency_graph_convolution_3_layer_call_fn_61445¤
²
FullArgSpec
args
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_61434¤
²
FullArgSpec
args
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_reshape_15_layer_call_fn_61464¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_15_layer_call_and_return_conditional_losses_61459¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_permute_3_layer_call_fn_58612à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
D__inference_permute_3_layer_call_and_return_conditional_losses_58606à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_reshape_16_layer_call_fn_61482¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_16_layer_call_and_return_conditional_losses_61477¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
&__inference_lstm_3_layer_call_fn_62122
&__inference_lstm_3_layer_call_fn_61791
&__inference_lstm_3_layer_call_fn_62111
&__inference_lstm_3_layer_call_fn_61802Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_lstm_3_layer_call_and_return_conditional_losses_61951
A__inference_lstm_3_layer_call_and_return_conditional_losses_61631
A__inference_lstm_3_layer_call_and_return_conditional_losses_62100
A__inference_lstm_3_layer_call_and_return_conditional_losses_61780Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_5_layer_call_fn_62149
)__inference_dropout_5_layer_call_fn_62144´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_5_layer_call_and_return_conditional_losses_62139
D__inference_dropout_5_layer_call_and_return_conditional_losses_62134´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_dense_7_layer_call_fn_62169¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_7_layer_call_and_return_conditional_losses_62160¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_lstm_cell_3_layer_call_fn_62265
+__inference_lstm_cell_3_layer_call_fn_62248¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_62200
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_62231¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ½
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60011v
/()*+,-.A¢>
7¢4
*'
input_11ÿÿÿÿÿÿÿÿÿF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ½
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60038v
/()*+,-.A¢>
7¢4
*'
input_11ÿÿÿÿÿÿÿÿÿF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 »
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60458t
/()*+,-.?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 »
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_60731t
/()*+,-.?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 
(__inference_T-GCN-WX_layer_call_fn_60091i
/()*+,-.A¢>
7¢4
*'
input_11ÿÿÿÿÿÿÿÿÿF
p

 
ª "ÿÿÿÿÿÿÿÿÿF
(__inference_T-GCN-WX_layer_call_fn_60143i
/()*+,-.A¢>
7¢4
*'
input_11ÿÿÿÿÿÿÿÿÿF
p 

 
ª "ÿÿÿÿÿÿÿÿÿF
(__inference_T-GCN-WX_layer_call_fn_60756g
/()*+,-.?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿF
p

 
ª "ÿÿÿÿÿÿÿÿÿF
(__inference_T-GCN-WX_layer_call_fn_60781g
/()*+,-.?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿF
p 

 
ª "ÿÿÿÿÿÿÿÿÿF
 __inference__wrapped_model_58599z
/()*+,-.9¢6
/¢,
*'
input_11ÿÿÿÿÿÿÿÿÿF
ª "1ª.
,
model_8!
model_8ÿÿÿÿÿÿÿÿÿF£
B__inference_dense_7_layer_call_and_return_conditional_losses_62160]-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 {
'__inference_dense_7_layer_call_fn_62169P-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿF²
B__inference_dense_8_layer_call_and_return_conditional_losses_60811l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿF
 
'__inference_dense_8_layer_call_fn_60820_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª " ÿÿÿÿÿÿÿÿÿF¦
D__inference_dropout_5_layer_call_and_return_conditional_losses_62134^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 ¦
D__inference_dropout_5_layer_call_and_return_conditional_losses_62139^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 ~
)__inference_dropout_5_layer_call_fn_62144Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p
ª "ÿÿÿÿÿÿÿÿÿÈ~
)__inference_dropout_5_layer_call_fn_62149Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p 
ª "ÿÿÿÿÿÿÿÿÿÈÉ
^__inference_fixed_adjacency_graph_convolution_3_layer_call_and_return_conditional_losses_61434g/()5¢2
+¢(
&#
featuresÿÿÿÿÿÿÿÿÿF
ª ")¢&

0ÿÿÿÿÿÿÿÿÿF

 ¡
C__inference_fixed_adjacency_graph_convolution_3_layer_call_fn_61445Z/()5¢2
+¢(
&#
featuresÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿF
³
A__inference_lstm_3_layer_call_and_return_conditional_losses_61631n*+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
F

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 ³
A__inference_lstm_3_layer_call_and_return_conditional_losses_61780n*+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
F

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 Ã
A__inference_lstm_3_layer_call_and_return_conditional_losses_61951~*+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 Ã
A__inference_lstm_3_layer_call_and_return_conditional_losses_62100~*+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 
&__inference_lstm_3_layer_call_fn_61791a*+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
F

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
&__inference_lstm_3_layer_call_fn_61802a*+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
F

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÈ
&__inference_lstm_3_layer_call_fn_62111q*+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
&__inference_lstm_3_layer_call_fn_62122q*+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÈÍ
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_62200*+,¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿF
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÈ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÈ
 
0/1/1ÿÿÿÿÿÿÿÿÿÈ
 Í
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_62231*+,¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿF
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÈ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÈ
 
0/1/1ÿÿÿÿÿÿÿÿÿÈ
 ¢
+__inference_lstm_cell_3_layer_call_fn_62248ò*+,¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿF
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÈ
C@

1/0ÿÿÿÿÿÿÿÿÿÈ

1/1ÿÿÿÿÿÿÿÿÿÈ¢
+__inference_lstm_cell_3_layer_call_fn_62265ò*+,¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿF
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÈ
C@

1/0ÿÿÿÿÿÿÿÿÿÈ

1/1ÿÿÿÿÿÿÿÿÿÈ¶
B__inference_model_8_layer_call_and_return_conditional_losses_59747p/()*+,-.=¢:
3¢0
&#
input_12ÿÿÿÿÿÿÿÿÿF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ¶
B__inference_model_8_layer_call_and_return_conditional_losses_59777p/()*+,-.=¢:
3¢0
&#
input_12ÿÿÿÿÿÿÿÿÿF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ´
B__inference_model_8_layer_call_and_return_conditional_losses_61083n/()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ´
B__inference_model_8_layer_call_and_return_conditional_losses_61321n/()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 
'__inference_model_8_layer_call_fn_59829c/()*+,-.=¢:
3¢0
&#
input_12ÿÿÿÿÿÿÿÿÿF
p

 
ª "ÿÿÿÿÿÿÿÿÿF
'__inference_model_8_layer_call_fn_59880c/()*+,-.=¢:
3¢0
&#
input_12ÿÿÿÿÿÿÿÿÿF
p 

 
ª "ÿÿÿÿÿÿÿÿÿF
'__inference_model_8_layer_call_fn_61342a/()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p

 
ª "ÿÿÿÿÿÿÿÿÿF
'__inference_model_8_layer_call_fn_61363a/()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p 

 
ª "ÿÿÿÿÿÿÿÿÿFç
D__inference_permute_3_layer_call_and_return_conditional_losses_58606R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
)__inference_permute_3_layer_call_fn_58612R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ­
E__inference_reshape_13_layer_call_and_return_conditional_losses_60833d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª ")¢&

0ÿÿÿÿÿÿÿÿÿF
 
*__inference_reshape_13_layer_call_fn_60838W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿF­
E__inference_reshape_14_layer_call_and_return_conditional_losses_61376d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª ")¢&

0ÿÿÿÿÿÿÿÿÿF
 
*__inference_reshape_14_layer_call_fn_61381W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿF­
E__inference_reshape_15_layer_call_and_return_conditional_losses_61459d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿF

ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿF

 
*__inference_reshape_15_layer_call_fn_61464W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿF

ª " ÿÿÿÿÿÿÿÿÿF
­
E__inference_reshape_16_layer_call_and_return_conditional_losses_61477d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
F
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
F
 
*__inference_reshape_16_layer_call_fn_61482W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
F
ª "ÿÿÿÿÿÿÿÿÿ
F®
#__inference_signature_wrapper_60178
/()*+,-.E¢B
¢ 
;ª8
6
input_11*'
input_11ÿÿÿÿÿÿÿÿÿF"1ª.
,
model_8!
model_8ÿÿÿÿÿÿÿÿÿF