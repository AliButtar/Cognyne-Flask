??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
?
conv1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1_1/kernel
y
"conv1_1/kernel/Read/ReadVariableOpReadVariableOpconv1_1/kernel*&
_output_shapes
:@*
dtype0
p
conv1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1_1/bias
i
 conv1_1/bias/Read/ReadVariableOpReadVariableOpconv1_1/bias*
_output_shapes
:@*
dtype0
?
conv1_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv1_2/kernel
y
"conv1_2/kernel/Read/ReadVariableOpReadVariableOpconv1_2/kernel*&
_output_shapes
:@@*
dtype0
p
conv1_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1_2/bias
i
 conv1_2/bias/Read/ReadVariableOpReadVariableOpconv1_2/bias*
_output_shapes
:@*
dtype0
?
conv2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameconv2_1/kernel
z
"conv2_1/kernel/Read/ReadVariableOpReadVariableOpconv2_1/kernel*'
_output_shapes
:@?*
dtype0
q
conv2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2_1/bias
j
 conv2_1/bias/Read/ReadVariableOpReadVariableOpconv2_1/bias*
_output_shapes	
:?*
dtype0
?
conv2_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv2_2/kernel
{
"conv2_2/kernel/Read/ReadVariableOpReadVariableOpconv2_2/kernel*(
_output_shapes
:??*
dtype0
q
conv2_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2_2/bias
j
 conv2_2/bias/Read/ReadVariableOpReadVariableOpconv2_2/bias*
_output_shapes	
:?*
dtype0
?
conv3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv3_1/kernel
{
"conv3_1/kernel/Read/ReadVariableOpReadVariableOpconv3_1/kernel*(
_output_shapes
:??*
dtype0
q
conv3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_1/bias
j
 conv3_1/bias/Read/ReadVariableOpReadVariableOpconv3_1/bias*
_output_shapes	
:?*
dtype0
?
conv3_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv3_2/kernel
{
"conv3_2/kernel/Read/ReadVariableOpReadVariableOpconv3_2/kernel*(
_output_shapes
:??*
dtype0
q
conv3_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_2/bias
j
 conv3_2/bias/Read/ReadVariableOpReadVariableOpconv3_2/bias*
_output_shapes	
:?*
dtype0
?
conv3_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv3_3/kernel
{
"conv3_3/kernel/Read/ReadVariableOpReadVariableOpconv3_3/kernel*(
_output_shapes
:??*
dtype0
q
conv3_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_3/bias
j
 conv3_3/bias/Read/ReadVariableOpReadVariableOpconv3_3/bias*
_output_shapes	
:?*
dtype0
?
conv4_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv4_1/kernel
{
"conv4_1/kernel/Read/ReadVariableOpReadVariableOpconv4_1/kernel*(
_output_shapes
:??*
dtype0
q
conv4_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv4_1/bias
j
 conv4_1/bias/Read/ReadVariableOpReadVariableOpconv4_1/bias*
_output_shapes	
:?*
dtype0
?
conv4_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv4_2/kernel
{
"conv4_2/kernel/Read/ReadVariableOpReadVariableOpconv4_2/kernel*(
_output_shapes
:??*
dtype0
q
conv4_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv4_2/bias
j
 conv4_2/bias/Read/ReadVariableOpReadVariableOpconv4_2/bias*
_output_shapes	
:?*
dtype0
?
conv4_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv4_3/kernel
{
"conv4_3/kernel/Read/ReadVariableOpReadVariableOpconv4_3/kernel*(
_output_shapes
:??*
dtype0
q
conv4_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv4_3/bias
j
 conv4_3/bias/Read/ReadVariableOpReadVariableOpconv4_3/bias*
_output_shapes	
:?*
dtype0
?
conv5_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv5_1/kernel
{
"conv5_1/kernel/Read/ReadVariableOpReadVariableOpconv5_1/kernel*(
_output_shapes
:??*
dtype0
q
conv5_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv5_1/bias
j
 conv5_1/bias/Read/ReadVariableOpReadVariableOpconv5_1/bias*
_output_shapes	
:?*
dtype0
?
conv5_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv5_2/kernel
{
"conv5_2/kernel/Read/ReadVariableOpReadVariableOpconv5_2/kernel*(
_output_shapes
:??*
dtype0
q
conv5_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv5_2/bias
j
 conv5_2/bias/Read/ReadVariableOpReadVariableOpconv5_2/bias*
_output_shapes	
:?*
dtype0
?
conv5_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv5_3/kernel
{
"conv5_3/kernel/Read/ReadVariableOpReadVariableOpconv5_3/kernel*(
_output_shapes
:??*
dtype0
q
conv5_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv5_3/bias
j
 conv5_3/bias/Read/ReadVariableOpReadVariableOpconv5_3/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?L
value?LB?L B?L
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer-19
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
R
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
R
6trainable_variables
7regularization_losses
8	variables
9	keras_api
h

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
h

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
R
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
h

Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
h

Vkernel
Wbias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
h

\kernel
]bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
R
btrainable_variables
cregularization_losses
d	variables
e	keras_api
h

fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
h

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
h

rkernel
sbias
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
R
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
R
|trainable_variables
}regularization_losses
~	variables
	keras_api
?
0
1
 2
!3
*4
+5
06
17
:8
;9
@10
A11
F12
G13
P14
Q15
V16
W17
\18
]19
f20
g21
l22
m23
r24
s25
 
?
0
1
 2
!3
*4
+5
06
17
:8
;9
@10
A11
F12
G13
P14
Q15
V16
W17
\18
]19
f20
g21
l22
m23
r24
s25
?
 ?layer_regularization_losses
trainable_variables
?layers
regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
	variables
 
ZX
VARIABLE_VALUEconv1_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
 ?layer_regularization_losses
trainable_variables
?layers
regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
	variables
ZX
VARIABLE_VALUEconv1_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?
 ?layer_regularization_losses
"trainable_variables
?layers
#regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
$	variables
 
 
 
?
 ?layer_regularization_losses
&trainable_variables
?layers
'regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
(	variables
ZX
VARIABLE_VALUEconv2_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
?
 ?layer_regularization_losses
,trainable_variables
?layers
-regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
.	variables
ZX
VARIABLE_VALUEconv2_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?
 ?layer_regularization_losses
2trainable_variables
?layers
3regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
4	variables
 
 
 
?
 ?layer_regularization_losses
6trainable_variables
?layers
7regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
8	variables
ZX
VARIABLE_VALUEconv3_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv3_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?
 ?layer_regularization_losses
<trainable_variables
?layers
=regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
>	variables
ZX
VARIABLE_VALUEconv3_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv3_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
?
 ?layer_regularization_losses
Btrainable_variables
?layers
Cregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
D	variables
ZX
VARIABLE_VALUEconv3_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv3_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 

F0
G1
?
 ?layer_regularization_losses
Htrainable_variables
?layers
Iregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
J	variables
 
 
 
?
 ?layer_regularization_losses
Ltrainable_variables
?layers
Mregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
N	variables
ZX
VARIABLE_VALUEconv4_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv4_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
 

P0
Q1
?
 ?layer_regularization_losses
Rtrainable_variables
?layers
Sregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
T	variables
ZX
VARIABLE_VALUEconv4_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv4_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
?
 ?layer_regularization_losses
Xtrainable_variables
?layers
Yregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
Z	variables
ZX
VARIABLE_VALUEconv4_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv4_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
 

\0
]1
?
 ?layer_regularization_losses
^trainable_variables
?layers
_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
`	variables
 
 
 
?
 ?layer_regularization_losses
btrainable_variables
?layers
cregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
d	variables
[Y
VARIABLE_VALUEconv5_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv5_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
?
 ?layer_regularization_losses
htrainable_variables
?layers
iregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
j	variables
[Y
VARIABLE_VALUEconv5_2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv5_2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
?
 ?layer_regularization_losses
ntrainable_variables
?layers
oregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
p	variables
[Y
VARIABLE_VALUEconv5_3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv5_3/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

r0
s1
 

r0
s1
?
 ?layer_regularization_losses
ttrainable_variables
?layers
uregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
v	variables
 
 
 
?
 ?layer_regularization_losses
xtrainable_variables
?layers
yregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
z	variables
 
 
 
?
 ?layer_regularization_losses
|trainable_variables
?layers
}regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
~	variables
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
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
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1_1/kernelconv1_1/biasconv1_2/kernelconv1_2/biasconv2_1/kernelconv2_1/biasconv2_2/kernelconv2_2/biasconv3_1/kernelconv3_1/biasconv3_2/kernelconv3_2/biasconv3_3/kernelconv3_3/biasconv4_1/kernelconv4_1/biasconv4_2/kernelconv4_2/biasconv4_3/kernelconv4_3/biasconv5_1/kernelconv5_1/biasconv5_2/kernelconv5_2/biasconv5_3/kernelconv5_3/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_1405
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"conv1_1/kernel/Read/ReadVariableOp conv1_1/bias/Read/ReadVariableOp"conv1_2/kernel/Read/ReadVariableOp conv1_2/bias/Read/ReadVariableOp"conv2_1/kernel/Read/ReadVariableOp conv2_1/bias/Read/ReadVariableOp"conv2_2/kernel/Read/ReadVariableOp conv2_2/bias/Read/ReadVariableOp"conv3_1/kernel/Read/ReadVariableOp conv3_1/bias/Read/ReadVariableOp"conv3_2/kernel/Read/ReadVariableOp conv3_2/bias/Read/ReadVariableOp"conv3_3/kernel/Read/ReadVariableOp conv3_3/bias/Read/ReadVariableOp"conv4_1/kernel/Read/ReadVariableOp conv4_1/bias/Read/ReadVariableOp"conv4_2/kernel/Read/ReadVariableOp conv4_2/bias/Read/ReadVariableOp"conv4_3/kernel/Read/ReadVariableOp conv4_3/bias/Read/ReadVariableOp"conv5_1/kernel/Read/ReadVariableOp conv5_1/bias/Read/ReadVariableOp"conv5_2/kernel/Read/ReadVariableOp conv5_2/bias/Read/ReadVariableOp"conv5_3/kernel/Read/ReadVariableOp conv5_3/bias/Read/ReadVariableOpConst*'
Tin 
2*
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
GPU 2J 8? *&
f!R
__inference__traced_save_2084
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1_1/kernelconv1_1/biasconv1_2/kernelconv1_2/biasconv2_1/kernelconv2_1/biasconv2_2/kernelconv2_2/biasconv3_1/kernelconv3_1/biasconv3_2/kernelconv3_2/biasconv3_3/kernelconv3_3/biasconv4_1/kernelconv4_1/biasconv4_2/kernelconv4_2/biasconv4_3/kernelconv4_3/biasconv5_1/kernelconv5_1/biasconv5_2/kernelconv5_2/biasconv5_3/kernelconv5_3/bias*&
Tin
2*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_2172??
ٳ
?
__inference__wrapped_model_457
input_1N
4vggface_vgg16_conv1_1_conv2d_readvariableop_resource:@C
5vggface_vgg16_conv1_1_biasadd_readvariableop_resource:@N
4vggface_vgg16_conv1_2_conv2d_readvariableop_resource:@@C
5vggface_vgg16_conv1_2_biasadd_readvariableop_resource:@O
4vggface_vgg16_conv2_1_conv2d_readvariableop_resource:@?D
5vggface_vgg16_conv2_1_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv2_2_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv2_2_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv3_1_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv3_1_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv3_2_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv3_2_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv3_3_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv3_3_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv4_1_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv4_1_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv4_2_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv4_2_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv4_3_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv4_3_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv5_1_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv5_1_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv5_2_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv5_2_biasadd_readvariableop_resource:	?P
4vggface_vgg16_conv5_3_conv2d_readvariableop_resource:??D
5vggface_vgg16_conv5_3_biasadd_readvariableop_resource:	?
identity??,vggface_vgg16/conv1_1/BiasAdd/ReadVariableOp?+vggface_vgg16/conv1_1/Conv2D/ReadVariableOp?,vggface_vgg16/conv1_2/BiasAdd/ReadVariableOp?+vggface_vgg16/conv1_2/Conv2D/ReadVariableOp?,vggface_vgg16/conv2_1/BiasAdd/ReadVariableOp?+vggface_vgg16/conv2_1/Conv2D/ReadVariableOp?,vggface_vgg16/conv2_2/BiasAdd/ReadVariableOp?+vggface_vgg16/conv2_2/Conv2D/ReadVariableOp?,vggface_vgg16/conv3_1/BiasAdd/ReadVariableOp?+vggface_vgg16/conv3_1/Conv2D/ReadVariableOp?,vggface_vgg16/conv3_2/BiasAdd/ReadVariableOp?+vggface_vgg16/conv3_2/Conv2D/ReadVariableOp?,vggface_vgg16/conv3_3/BiasAdd/ReadVariableOp?+vggface_vgg16/conv3_3/Conv2D/ReadVariableOp?,vggface_vgg16/conv4_1/BiasAdd/ReadVariableOp?+vggface_vgg16/conv4_1/Conv2D/ReadVariableOp?,vggface_vgg16/conv4_2/BiasAdd/ReadVariableOp?+vggface_vgg16/conv4_2/Conv2D/ReadVariableOp?,vggface_vgg16/conv4_3/BiasAdd/ReadVariableOp?+vggface_vgg16/conv4_3/Conv2D/ReadVariableOp?,vggface_vgg16/conv5_1/BiasAdd/ReadVariableOp?+vggface_vgg16/conv5_1/Conv2D/ReadVariableOp?,vggface_vgg16/conv5_2/BiasAdd/ReadVariableOp?+vggface_vgg16/conv5_2/Conv2D/ReadVariableOp?,vggface_vgg16/conv5_3/BiasAdd/ReadVariableOp?+vggface_vgg16/conv5_3/Conv2D/ReadVariableOp?
+vggface_vgg16/conv1_1/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+vggface_vgg16/conv1_1/Conv2D/ReadVariableOp?
vggface_vgg16/conv1_1/Conv2DConv2Dinput_13vggface_vgg16/conv1_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vggface_vgg16/conv1_1/Conv2D?
,vggface_vgg16/conv1_1/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,vggface_vgg16/conv1_1/BiasAdd/ReadVariableOp?
vggface_vgg16/conv1_1/BiasAddBiasAdd%vggface_vgg16/conv1_1/Conv2D:output:04vggface_vgg16/conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vggface_vgg16/conv1_1/BiasAdd?
vggface_vgg16/conv1_1/ReluRelu&vggface_vgg16/conv1_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vggface_vgg16/conv1_1/Relu?
+vggface_vgg16/conv1_2/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv1_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+vggface_vgg16/conv1_2/Conv2D/ReadVariableOp?
vggface_vgg16/conv1_2/Conv2DConv2D(vggface_vgg16/conv1_1/Relu:activations:03vggface_vgg16/conv1_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vggface_vgg16/conv1_2/Conv2D?
,vggface_vgg16/conv1_2/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv1_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,vggface_vgg16/conv1_2/BiasAdd/ReadVariableOp?
vggface_vgg16/conv1_2/BiasAddBiasAdd%vggface_vgg16/conv1_2/Conv2D:output:04vggface_vgg16/conv1_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vggface_vgg16/conv1_2/BiasAdd?
vggface_vgg16/conv1_2/ReluRelu&vggface_vgg16/conv1_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vggface_vgg16/conv1_2/Relu?
vggface_vgg16/pool1/MaxPoolMaxPool(vggface_vgg16/conv1_2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
vggface_vgg16/pool1/MaxPool?
+vggface_vgg16/conv2_1/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+vggface_vgg16/conv2_1/Conv2D/ReadVariableOp?
vggface_vgg16/conv2_1/Conv2DConv2D$vggface_vgg16/pool1/MaxPool:output:03vggface_vgg16/conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vggface_vgg16/conv2_1/Conv2D?
,vggface_vgg16/conv2_1/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv2_1/BiasAdd/ReadVariableOp?
vggface_vgg16/conv2_1/BiasAddBiasAdd%vggface_vgg16/conv2_1/Conv2D:output:04vggface_vgg16/conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vggface_vgg16/conv2_1/BiasAdd?
vggface_vgg16/conv2_1/ReluRelu&vggface_vgg16/conv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vggface_vgg16/conv2_1/Relu?
+vggface_vgg16/conv2_2/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv2_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv2_2/Conv2D/ReadVariableOp?
vggface_vgg16/conv2_2/Conv2DConv2D(vggface_vgg16/conv2_1/Relu:activations:03vggface_vgg16/conv2_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vggface_vgg16/conv2_2/Conv2D?
,vggface_vgg16/conv2_2/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv2_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv2_2/BiasAdd/ReadVariableOp?
vggface_vgg16/conv2_2/BiasAddBiasAdd%vggface_vgg16/conv2_2/Conv2D:output:04vggface_vgg16/conv2_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vggface_vgg16/conv2_2/BiasAdd?
vggface_vgg16/conv2_2/ReluRelu&vggface_vgg16/conv2_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vggface_vgg16/conv2_2/Relu?
vggface_vgg16/pool2/MaxPoolMaxPool(vggface_vgg16/conv2_2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
vggface_vgg16/pool2/MaxPool?
+vggface_vgg16/conv3_1/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv3_1/Conv2D/ReadVariableOp?
vggface_vgg16/conv3_1/Conv2DConv2D$vggface_vgg16/pool2/MaxPool:output:03vggface_vgg16/conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vggface_vgg16/conv3_1/Conv2D?
,vggface_vgg16/conv3_1/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv3_1/BiasAdd/ReadVariableOp?
vggface_vgg16/conv3_1/BiasAddBiasAdd%vggface_vgg16/conv3_1/Conv2D:output:04vggface_vgg16/conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vggface_vgg16/conv3_1/BiasAdd?
vggface_vgg16/conv3_1/ReluRelu&vggface_vgg16/conv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vggface_vgg16/conv3_1/Relu?
+vggface_vgg16/conv3_2/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv3_2/Conv2D/ReadVariableOp?
vggface_vgg16/conv3_2/Conv2DConv2D(vggface_vgg16/conv3_1/Relu:activations:03vggface_vgg16/conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vggface_vgg16/conv3_2/Conv2D?
,vggface_vgg16/conv3_2/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv3_2/BiasAdd/ReadVariableOp?
vggface_vgg16/conv3_2/BiasAddBiasAdd%vggface_vgg16/conv3_2/Conv2D:output:04vggface_vgg16/conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vggface_vgg16/conv3_2/BiasAdd?
vggface_vgg16/conv3_2/ReluRelu&vggface_vgg16/conv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vggface_vgg16/conv3_2/Relu?
+vggface_vgg16/conv3_3/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv3_3/Conv2D/ReadVariableOp?
vggface_vgg16/conv3_3/Conv2DConv2D(vggface_vgg16/conv3_2/Relu:activations:03vggface_vgg16/conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vggface_vgg16/conv3_3/Conv2D?
,vggface_vgg16/conv3_3/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv3_3/BiasAdd/ReadVariableOp?
vggface_vgg16/conv3_3/BiasAddBiasAdd%vggface_vgg16/conv3_3/Conv2D:output:04vggface_vgg16/conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vggface_vgg16/conv3_3/BiasAdd?
vggface_vgg16/conv3_3/ReluRelu&vggface_vgg16/conv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vggface_vgg16/conv3_3/Relu?
vggface_vgg16/pool3/MaxPoolMaxPool(vggface_vgg16/conv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vggface_vgg16/pool3/MaxPool?
+vggface_vgg16/conv4_1/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv4_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv4_1/Conv2D/ReadVariableOp?
vggface_vgg16/conv4_1/Conv2DConv2D$vggface_vgg16/pool3/MaxPool:output:03vggface_vgg16/conv4_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vggface_vgg16/conv4_1/Conv2D?
,vggface_vgg16/conv4_1/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv4_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv4_1/BiasAdd/ReadVariableOp?
vggface_vgg16/conv4_1/BiasAddBiasAdd%vggface_vgg16/conv4_1/Conv2D:output:04vggface_vgg16/conv4_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv4_1/BiasAdd?
vggface_vgg16/conv4_1/ReluRelu&vggface_vgg16/conv4_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv4_1/Relu?
+vggface_vgg16/conv4_2/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv4_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv4_2/Conv2D/ReadVariableOp?
vggface_vgg16/conv4_2/Conv2DConv2D(vggface_vgg16/conv4_1/Relu:activations:03vggface_vgg16/conv4_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vggface_vgg16/conv4_2/Conv2D?
,vggface_vgg16/conv4_2/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv4_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv4_2/BiasAdd/ReadVariableOp?
vggface_vgg16/conv4_2/BiasAddBiasAdd%vggface_vgg16/conv4_2/Conv2D:output:04vggface_vgg16/conv4_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv4_2/BiasAdd?
vggface_vgg16/conv4_2/ReluRelu&vggface_vgg16/conv4_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv4_2/Relu?
+vggface_vgg16/conv4_3/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv4_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv4_3/Conv2D/ReadVariableOp?
vggface_vgg16/conv4_3/Conv2DConv2D(vggface_vgg16/conv4_2/Relu:activations:03vggface_vgg16/conv4_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vggface_vgg16/conv4_3/Conv2D?
,vggface_vgg16/conv4_3/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv4_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv4_3/BiasAdd/ReadVariableOp?
vggface_vgg16/conv4_3/BiasAddBiasAdd%vggface_vgg16/conv4_3/Conv2D:output:04vggface_vgg16/conv4_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv4_3/BiasAdd?
vggface_vgg16/conv4_3/ReluRelu&vggface_vgg16/conv4_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv4_3/Relu?
vggface_vgg16/pool4/MaxPoolMaxPool(vggface_vgg16/conv4_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vggface_vgg16/pool4/MaxPool?
+vggface_vgg16/conv5_1/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv5_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv5_1/Conv2D/ReadVariableOp?
vggface_vgg16/conv5_1/Conv2DConv2D$vggface_vgg16/pool4/MaxPool:output:03vggface_vgg16/conv5_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vggface_vgg16/conv5_1/Conv2D?
,vggface_vgg16/conv5_1/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv5_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv5_1/BiasAdd/ReadVariableOp?
vggface_vgg16/conv5_1/BiasAddBiasAdd%vggface_vgg16/conv5_1/Conv2D:output:04vggface_vgg16/conv5_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv5_1/BiasAdd?
vggface_vgg16/conv5_1/ReluRelu&vggface_vgg16/conv5_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv5_1/Relu?
+vggface_vgg16/conv5_2/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv5_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv5_2/Conv2D/ReadVariableOp?
vggface_vgg16/conv5_2/Conv2DConv2D(vggface_vgg16/conv5_1/Relu:activations:03vggface_vgg16/conv5_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vggface_vgg16/conv5_2/Conv2D?
,vggface_vgg16/conv5_2/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv5_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv5_2/BiasAdd/ReadVariableOp?
vggface_vgg16/conv5_2/BiasAddBiasAdd%vggface_vgg16/conv5_2/Conv2D:output:04vggface_vgg16/conv5_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv5_2/BiasAdd?
vggface_vgg16/conv5_2/ReluRelu&vggface_vgg16/conv5_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv5_2/Relu?
+vggface_vgg16/conv5_3/Conv2D/ReadVariableOpReadVariableOp4vggface_vgg16_conv5_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+vggface_vgg16/conv5_3/Conv2D/ReadVariableOp?
vggface_vgg16/conv5_3/Conv2DConv2D(vggface_vgg16/conv5_2/Relu:activations:03vggface_vgg16/conv5_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vggface_vgg16/conv5_3/Conv2D?
,vggface_vgg16/conv5_3/BiasAdd/ReadVariableOpReadVariableOp5vggface_vgg16_conv5_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,vggface_vgg16/conv5_3/BiasAdd/ReadVariableOp?
vggface_vgg16/conv5_3/BiasAddBiasAdd%vggface_vgg16/conv5_3/Conv2D:output:04vggface_vgg16/conv5_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv5_3/BiasAdd?
vggface_vgg16/conv5_3/ReluRelu&vggface_vgg16/conv5_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vggface_vgg16/conv5_3/Relu?
vggface_vgg16/pool5/MaxPoolMaxPool(vggface_vgg16/conv5_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vggface_vgg16/pool5/MaxPool?
=vggface_vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2?
=vggface_vgg16/global_average_pooling2d/Mean/reduction_indices?
+vggface_vgg16/global_average_pooling2d/MeanMean$vggface_vgg16/pool5/MaxPool:output:0Fvggface_vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2-
+vggface_vgg16/global_average_pooling2d/Mean?

IdentityIdentity4vggface_vgg16/global_average_pooling2d/Mean:output:0-^vggface_vgg16/conv1_1/BiasAdd/ReadVariableOp,^vggface_vgg16/conv1_1/Conv2D/ReadVariableOp-^vggface_vgg16/conv1_2/BiasAdd/ReadVariableOp,^vggface_vgg16/conv1_2/Conv2D/ReadVariableOp-^vggface_vgg16/conv2_1/BiasAdd/ReadVariableOp,^vggface_vgg16/conv2_1/Conv2D/ReadVariableOp-^vggface_vgg16/conv2_2/BiasAdd/ReadVariableOp,^vggface_vgg16/conv2_2/Conv2D/ReadVariableOp-^vggface_vgg16/conv3_1/BiasAdd/ReadVariableOp,^vggface_vgg16/conv3_1/Conv2D/ReadVariableOp-^vggface_vgg16/conv3_2/BiasAdd/ReadVariableOp,^vggface_vgg16/conv3_2/Conv2D/ReadVariableOp-^vggface_vgg16/conv3_3/BiasAdd/ReadVariableOp,^vggface_vgg16/conv3_3/Conv2D/ReadVariableOp-^vggface_vgg16/conv4_1/BiasAdd/ReadVariableOp,^vggface_vgg16/conv4_1/Conv2D/ReadVariableOp-^vggface_vgg16/conv4_2/BiasAdd/ReadVariableOp,^vggface_vgg16/conv4_2/Conv2D/ReadVariableOp-^vggface_vgg16/conv4_3/BiasAdd/ReadVariableOp,^vggface_vgg16/conv4_3/Conv2D/ReadVariableOp-^vggface_vgg16/conv5_1/BiasAdd/ReadVariableOp,^vggface_vgg16/conv5_1/Conv2D/ReadVariableOp-^vggface_vgg16/conv5_2/BiasAdd/ReadVariableOp,^vggface_vgg16/conv5_2/Conv2D/ReadVariableOp-^vggface_vgg16/conv5_3/BiasAdd/ReadVariableOp,^vggface_vgg16/conv5_3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,vggface_vgg16/conv1_1/BiasAdd/ReadVariableOp,vggface_vgg16/conv1_1/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv1_1/Conv2D/ReadVariableOp+vggface_vgg16/conv1_1/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv1_2/BiasAdd/ReadVariableOp,vggface_vgg16/conv1_2/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv1_2/Conv2D/ReadVariableOp+vggface_vgg16/conv1_2/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv2_1/BiasAdd/ReadVariableOp,vggface_vgg16/conv2_1/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv2_1/Conv2D/ReadVariableOp+vggface_vgg16/conv2_1/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv2_2/BiasAdd/ReadVariableOp,vggface_vgg16/conv2_2/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv2_2/Conv2D/ReadVariableOp+vggface_vgg16/conv2_2/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv3_1/BiasAdd/ReadVariableOp,vggface_vgg16/conv3_1/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv3_1/Conv2D/ReadVariableOp+vggface_vgg16/conv3_1/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv3_2/BiasAdd/ReadVariableOp,vggface_vgg16/conv3_2/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv3_2/Conv2D/ReadVariableOp+vggface_vgg16/conv3_2/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv3_3/BiasAdd/ReadVariableOp,vggface_vgg16/conv3_3/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv3_3/Conv2D/ReadVariableOp+vggface_vgg16/conv3_3/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv4_1/BiasAdd/ReadVariableOp,vggface_vgg16/conv4_1/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv4_1/Conv2D/ReadVariableOp+vggface_vgg16/conv4_1/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv4_2/BiasAdd/ReadVariableOp,vggface_vgg16/conv4_2/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv4_2/Conv2D/ReadVariableOp+vggface_vgg16/conv4_2/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv4_3/BiasAdd/ReadVariableOp,vggface_vgg16/conv4_3/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv4_3/Conv2D/ReadVariableOp+vggface_vgg16/conv4_3/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv5_1/BiasAdd/ReadVariableOp,vggface_vgg16/conv5_1/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv5_1/Conv2D/ReadVariableOp+vggface_vgg16/conv5_1/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv5_2/BiasAdd/ReadVariableOp,vggface_vgg16/conv5_2/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv5_2/Conv2D/ReadVariableOp+vggface_vgg16/conv5_2/Conv2D/ReadVariableOp2\
,vggface_vgg16/conv5_3/BiasAdd/ReadVariableOp,vggface_vgg16/conv5_3/BiasAdd/ReadVariableOp2Z
+vggface_vgg16/conv5_3/Conv2D/ReadVariableOp+vggface_vgg16/conv5_3/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
@__inference_conv4_1_layer_call_and_return_conditional_losses_670

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_conv3_2_layer_call_and_return_conditional_losses_635

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
A__inference_conv3_3_layer_call_and_return_conditional_losses_1863

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
A__inference_conv4_2_layer_call_and_return_conditional_losses_1903

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_conv5_3_layer_call_fn_1972

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_3_layer_call_and_return_conditional_losses_7562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_conv2_2_layer_call_and_return_conditional_losses_600

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?
@__inference_conv4_3_layer_call_and_return_conditional_losses_704

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1621

inputs@
&conv1_1_conv2d_readvariableop_resource:@5
'conv1_1_biasadd_readvariableop_resource:@@
&conv1_2_conv2d_readvariableop_resource:@@5
'conv1_2_biasadd_readvariableop_resource:@A
&conv2_1_conv2d_readvariableop_resource:@?6
'conv2_1_biasadd_readvariableop_resource:	?B
&conv2_2_conv2d_readvariableop_resource:??6
'conv2_2_biasadd_readvariableop_resource:	?B
&conv3_1_conv2d_readvariableop_resource:??6
'conv3_1_biasadd_readvariableop_resource:	?B
&conv3_2_conv2d_readvariableop_resource:??6
'conv3_2_biasadd_readvariableop_resource:	?B
&conv3_3_conv2d_readvariableop_resource:??6
'conv3_3_biasadd_readvariableop_resource:	?B
&conv4_1_conv2d_readvariableop_resource:??6
'conv4_1_biasadd_readvariableop_resource:	?B
&conv4_2_conv2d_readvariableop_resource:??6
'conv4_2_biasadd_readvariableop_resource:	?B
&conv4_3_conv2d_readvariableop_resource:??6
'conv4_3_biasadd_readvariableop_resource:	?B
&conv5_1_conv2d_readvariableop_resource:??6
'conv5_1_biasadd_readvariableop_resource:	?B
&conv5_2_conv2d_readvariableop_resource:??6
'conv5_2_biasadd_readvariableop_resource:	?B
&conv5_3_conv2d_readvariableop_resource:??6
'conv5_3_biasadd_readvariableop_resource:	?
identity??conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?conv1_2/BiasAdd/ReadVariableOp?conv1_2/Conv2D/ReadVariableOp?conv2_1/BiasAdd/ReadVariableOp?conv2_1/Conv2D/ReadVariableOp?conv2_2/BiasAdd/ReadVariableOp?conv2_2/Conv2D/ReadVariableOp?conv3_1/BiasAdd/ReadVariableOp?conv3_1/Conv2D/ReadVariableOp?conv3_2/BiasAdd/ReadVariableOp?conv3_2/Conv2D/ReadVariableOp?conv3_3/BiasAdd/ReadVariableOp?conv3_3/Conv2D/ReadVariableOp?conv4_1/BiasAdd/ReadVariableOp?conv4_1/Conv2D/ReadVariableOp?conv4_2/BiasAdd/ReadVariableOp?conv4_2/Conv2D/ReadVariableOp?conv4_3/BiasAdd/ReadVariableOp?conv4_3/Conv2D/ReadVariableOp?conv5_1/BiasAdd/ReadVariableOp?conv5_1/Conv2D/ReadVariableOp?conv5_2/BiasAdd/ReadVariableOp?conv5_2/Conv2D/ReadVariableOp?conv5_3/BiasAdd/ReadVariableOp?conv5_3/Conv2D/ReadVariableOp?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1_1/Conv2D/ReadVariableOp?
conv1_1/Conv2DConv2Dinputs%conv1_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv1_1/Conv2D?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv1_1/BiasAdd/ReadVariableOp?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv1_1/BiasAddz
conv1_1/ReluReluconv1_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv1_1/Relu?
conv1_2/Conv2D/ReadVariableOpReadVariableOp&conv1_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv1_2/Conv2D/ReadVariableOp?
conv1_2/Conv2DConv2Dconv1_1/Relu:activations:0%conv1_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv1_2/Conv2D?
conv1_2/BiasAdd/ReadVariableOpReadVariableOp'conv1_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv1_2/BiasAdd/ReadVariableOp?
conv1_2/BiasAddBiasAddconv1_2/Conv2D:output:0&conv1_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv1_2/BiasAddz
conv1_2/ReluReluconv1_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv1_2/Relu?
pool1/MaxPoolMaxPoolconv1_2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
pool1/MaxPool?
conv2_1/Conv2D/ReadVariableOpReadVariableOp&conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv2_1/Conv2D/ReadVariableOp?
conv2_1/Conv2DConv2Dpool1/MaxPool:output:0%conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
conv2_1/Conv2D?
conv2_1/BiasAdd/ReadVariableOpReadVariableOp'conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv2_1/BiasAdd/ReadVariableOp?
conv2_1/BiasAddBiasAddconv2_1/Conv2D:output:0&conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
conv2_1/BiasAddy
conv2_1/ReluReluconv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
conv2_1/Relu?
conv2_2/Conv2D/ReadVariableOpReadVariableOp&conv2_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2_2/Conv2D/ReadVariableOp?
conv2_2/Conv2DConv2Dconv2_1/Relu:activations:0%conv2_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
conv2_2/Conv2D?
conv2_2/BiasAdd/ReadVariableOpReadVariableOp'conv2_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv2_2/BiasAdd/ReadVariableOp?
conv2_2/BiasAddBiasAddconv2_2/Conv2D:output:0&conv2_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
conv2_2/BiasAddy
conv2_2/ReluReluconv2_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
conv2_2/Relu?
pool2/MaxPoolMaxPoolconv2_2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
pool2/MaxPool?
conv3_1/Conv2D/ReadVariableOpReadVariableOp&conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_1/Conv2D/ReadVariableOp?
conv3_1/Conv2DConv2Dpool2/MaxPool:output:0%conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
conv3_1/Conv2D?
conv3_1/BiasAdd/ReadVariableOpReadVariableOp'conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_1/BiasAdd/ReadVariableOp?
conv3_1/BiasAddBiasAddconv3_1/Conv2D:output:0&conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
conv3_1/BiasAddy
conv3_1/ReluReluconv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
conv3_1/Relu?
conv3_2/Conv2D/ReadVariableOpReadVariableOp&conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_2/Conv2D/ReadVariableOp?
conv3_2/Conv2DConv2Dconv3_1/Relu:activations:0%conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
conv3_2/Conv2D?
conv3_2/BiasAdd/ReadVariableOpReadVariableOp'conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_2/BiasAdd/ReadVariableOp?
conv3_2/BiasAddBiasAddconv3_2/Conv2D:output:0&conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
conv3_2/BiasAddy
conv3_2/ReluReluconv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
conv3_2/Relu?
conv3_3/Conv2D/ReadVariableOpReadVariableOp&conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_3/Conv2D/ReadVariableOp?
conv3_3/Conv2DConv2Dconv3_2/Relu:activations:0%conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
conv3_3/Conv2D?
conv3_3/BiasAdd/ReadVariableOpReadVariableOp'conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_3/BiasAdd/ReadVariableOp?
conv3_3/BiasAddBiasAddconv3_3/Conv2D:output:0&conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
conv3_3/BiasAddy
conv3_3/ReluReluconv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
conv3_3/Relu?
pool3/MaxPoolMaxPoolconv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
pool3/MaxPool?
conv4_1/Conv2D/ReadVariableOpReadVariableOp&conv4_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_1/Conv2D/ReadVariableOp?
conv4_1/Conv2DConv2Dpool3/MaxPool:output:0%conv4_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_1/Conv2D?
conv4_1/BiasAdd/ReadVariableOpReadVariableOp'conv4_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_1/BiasAdd/ReadVariableOp?
conv4_1/BiasAddBiasAddconv4_1/Conv2D:output:0&conv4_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_1/BiasAddy
conv4_1/ReluReluconv4_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_1/Relu?
conv4_2/Conv2D/ReadVariableOpReadVariableOp&conv4_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_2/Conv2D/ReadVariableOp?
conv4_2/Conv2DConv2Dconv4_1/Relu:activations:0%conv4_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_2/Conv2D?
conv4_2/BiasAdd/ReadVariableOpReadVariableOp'conv4_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_2/BiasAdd/ReadVariableOp?
conv4_2/BiasAddBiasAddconv4_2/Conv2D:output:0&conv4_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_2/BiasAddy
conv4_2/ReluReluconv4_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_2/Relu?
conv4_3/Conv2D/ReadVariableOpReadVariableOp&conv4_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_3/Conv2D/ReadVariableOp?
conv4_3/Conv2DConv2Dconv4_2/Relu:activations:0%conv4_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_3/Conv2D?
conv4_3/BiasAdd/ReadVariableOpReadVariableOp'conv4_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_3/BiasAdd/ReadVariableOp?
conv4_3/BiasAddBiasAddconv4_3/Conv2D:output:0&conv4_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_3/BiasAddy
conv4_3/ReluReluconv4_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_3/Relu?
pool4/MaxPoolMaxPoolconv4_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
pool4/MaxPool?
conv5_1/Conv2D/ReadVariableOpReadVariableOp&conv5_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5_1/Conv2D/ReadVariableOp?
conv5_1/Conv2DConv2Dpool4/MaxPool:output:0%conv5_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv5_1/Conv2D?
conv5_1/BiasAdd/ReadVariableOpReadVariableOp'conv5_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv5_1/BiasAdd/ReadVariableOp?
conv5_1/BiasAddBiasAddconv5_1/Conv2D:output:0&conv5_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5_1/BiasAddy
conv5_1/ReluReluconv5_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv5_1/Relu?
conv5_2/Conv2D/ReadVariableOpReadVariableOp&conv5_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5_2/Conv2D/ReadVariableOp?
conv5_2/Conv2DConv2Dconv5_1/Relu:activations:0%conv5_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv5_2/Conv2D?
conv5_2/BiasAdd/ReadVariableOpReadVariableOp'conv5_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv5_2/BiasAdd/ReadVariableOp?
conv5_2/BiasAddBiasAddconv5_2/Conv2D:output:0&conv5_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5_2/BiasAddy
conv5_2/ReluReluconv5_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv5_2/Relu?
conv5_3/Conv2D/ReadVariableOpReadVariableOp&conv5_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5_3/Conv2D/ReadVariableOp?
conv5_3/Conv2DConv2Dconv5_2/Relu:activations:0%conv5_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv5_3/Conv2D?
conv5_3/BiasAdd/ReadVariableOpReadVariableOp'conv5_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv5_3/BiasAdd/ReadVariableOp?
conv5_3/BiasAddBiasAddconv5_3/Conv2D:output:0&conv5_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5_3/BiasAddy
conv5_3/ReluReluconv5_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv5_3/Relu?
pool5/MaxPoolMaxPoolconv5_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
pool5/MaxPool?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeanpool5/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling2d/Mean?
IdentityIdentity&global_average_pooling2d/Mean:output:0^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp^conv1_2/BiasAdd/ReadVariableOp^conv1_2/Conv2D/ReadVariableOp^conv2_1/BiasAdd/ReadVariableOp^conv2_1/Conv2D/ReadVariableOp^conv2_2/BiasAdd/ReadVariableOp^conv2_2/Conv2D/ReadVariableOp^conv3_1/BiasAdd/ReadVariableOp^conv3_1/Conv2D/ReadVariableOp^conv3_2/BiasAdd/ReadVariableOp^conv3_2/Conv2D/ReadVariableOp^conv3_3/BiasAdd/ReadVariableOp^conv3_3/Conv2D/ReadVariableOp^conv4_1/BiasAdd/ReadVariableOp^conv4_1/Conv2D/ReadVariableOp^conv4_2/BiasAdd/ReadVariableOp^conv4_2/Conv2D/ReadVariableOp^conv4_3/BiasAdd/ReadVariableOp^conv4_3/Conv2D/ReadVariableOp^conv5_1/BiasAdd/ReadVariableOp^conv5_1/Conv2D/ReadVariableOp^conv5_2/BiasAdd/ReadVariableOp^conv5_2/Conv2D/ReadVariableOp^conv5_3/BiasAdd/ReadVariableOp^conv5_3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2@
conv1_2/BiasAdd/ReadVariableOpconv1_2/BiasAdd/ReadVariableOp2>
conv1_2/Conv2D/ReadVariableOpconv1_2/Conv2D/ReadVariableOp2@
conv2_1/BiasAdd/ReadVariableOpconv2_1/BiasAdd/ReadVariableOp2>
conv2_1/Conv2D/ReadVariableOpconv2_1/Conv2D/ReadVariableOp2@
conv2_2/BiasAdd/ReadVariableOpconv2_2/BiasAdd/ReadVariableOp2>
conv2_2/Conv2D/ReadVariableOpconv2_2/Conv2D/ReadVariableOp2@
conv3_1/BiasAdd/ReadVariableOpconv3_1/BiasAdd/ReadVariableOp2>
conv3_1/Conv2D/ReadVariableOpconv3_1/Conv2D/ReadVariableOp2@
conv3_2/BiasAdd/ReadVariableOpconv3_2/BiasAdd/ReadVariableOp2>
conv3_2/Conv2D/ReadVariableOpconv3_2/Conv2D/ReadVariableOp2@
conv3_3/BiasAdd/ReadVariableOpconv3_3/BiasAdd/ReadVariableOp2>
conv3_3/Conv2D/ReadVariableOpconv3_3/Conv2D/ReadVariableOp2@
conv4_1/BiasAdd/ReadVariableOpconv4_1/BiasAdd/ReadVariableOp2>
conv4_1/Conv2D/ReadVariableOpconv4_1/Conv2D/ReadVariableOp2@
conv4_2/BiasAdd/ReadVariableOpconv4_2/BiasAdd/ReadVariableOp2>
conv4_2/Conv2D/ReadVariableOpconv4_2/Conv2D/ReadVariableOp2@
conv4_3/BiasAdd/ReadVariableOpconv4_3/BiasAdd/ReadVariableOp2>
conv4_3/Conv2D/ReadVariableOpconv4_3/Conv2D/ReadVariableOp2@
conv5_1/BiasAdd/ReadVariableOpconv5_1/BiasAdd/ReadVariableOp2>
conv5_1/Conv2D/ReadVariableOpconv5_1/Conv2D/ReadVariableOp2@
conv5_2/BiasAdd/ReadVariableOpconv5_2/BiasAdd/ReadVariableOp2>
conv5_2/Conv2D/ReadVariableOpconv5_2/Conv2D/ReadVariableOp2@
conv5_3/BiasAdd/ReadVariableOpconv5_3/BiasAdd/ReadVariableOp2>
conv5_3/Conv2D/ReadVariableOpconv5_3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
#__inference_pool3_layer_call_fn_493

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_4872
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
Z
>__inference_pool2_layer_call_and_return_conditional_losses_475

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
Z
>__inference_pool3_layer_call_and_return_conditional_losses_487

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_conv3_3_layer_call_fn_1852

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_3_layer_call_and_return_conditional_losses_6522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
@__inference_conv5_1_layer_call_and_return_conditional_losses_722

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?V
?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1271
input_1&
conv1_1_1199:@
conv1_1_1201:@&
conv1_2_1204:@@
conv1_2_1206:@'
conv2_1_1210:@?
conv2_1_1212:	?(
conv2_2_1215:??
conv2_2_1217:	?(
conv3_1_1221:??
conv3_1_1223:	?(
conv3_2_1226:??
conv3_2_1228:	?(
conv3_3_1231:??
conv3_3_1233:	?(
conv4_1_1237:??
conv4_1_1239:	?(
conv4_2_1242:??
conv4_2_1244:	?(
conv4_3_1247:??
conv4_3_1249:	?(
conv5_1_1253:??
conv5_1_1255:	?(
conv5_2_1258:??
conv5_2_1260:	?(
conv5_3_1263:??
conv5_3_1265:	?
identity??conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv2_1/StatefulPartitionedCall?conv2_2/StatefulPartitionedCall?conv3_1/StatefulPartitionedCall?conv3_2/StatefulPartitionedCall?conv3_3/StatefulPartitionedCall?conv4_1/StatefulPartitionedCall?conv4_2/StatefulPartitionedCall?conv4_3/StatefulPartitionedCall?conv5_1/StatefulPartitionedCall?conv5_2/StatefulPartitionedCall?conv5_3/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_1_1199conv1_1_1201*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_1_layer_call_and_return_conditional_losses_5482!
conv1_1/StatefulPartitionedCall?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0conv1_2_1204conv1_2_1206*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_2_layer_call_and_return_conditional_losses_5652!
conv1_2/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_4632
pool1/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_1_1210conv2_1_1212*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_1_layer_call_and_return_conditional_losses_5832!
conv2_1/StatefulPartitionedCall?
conv2_2/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0conv2_2_1215conv2_2_1217*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_2_layer_call_and_return_conditional_losses_6002!
conv2_2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall(conv2_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_4752
pool2/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_1_1221conv3_1_1223*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_1_layer_call_and_return_conditional_losses_6182!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_1226conv3_2_1228*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_2_layer_call_and_return_conditional_losses_6352!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_1231conv3_3_1233*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_3_layer_call_and_return_conditional_losses_6522!
conv3_3/StatefulPartitionedCall?
pool3/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_4872
pool3/PartitionedCall?
conv4_1/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_1_1237conv4_1_1239*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_1_layer_call_and_return_conditional_losses_6702!
conv4_1/StatefulPartitionedCall?
conv4_2/StatefulPartitionedCallStatefulPartitionedCall(conv4_1/StatefulPartitionedCall:output:0conv4_2_1242conv4_2_1244*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_2_layer_call_and_return_conditional_losses_6872!
conv4_2/StatefulPartitionedCall?
conv4_3/StatefulPartitionedCallStatefulPartitionedCall(conv4_2/StatefulPartitionedCall:output:0conv4_3_1247conv4_3_1249*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_3_layer_call_and_return_conditional_losses_7042!
conv4_3/StatefulPartitionedCall?
pool4/PartitionedCallPartitionedCall(conv4_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool4_layer_call_and_return_conditional_losses_4992
pool4/PartitionedCall?
conv5_1/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0conv5_1_1253conv5_1_1255*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_1_layer_call_and_return_conditional_losses_7222!
conv5_1/StatefulPartitionedCall?
conv5_2/StatefulPartitionedCallStatefulPartitionedCall(conv5_1/StatefulPartitionedCall:output:0conv5_2_1258conv5_2_1260*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_2_layer_call_and_return_conditional_losses_7392!
conv5_2/StatefulPartitionedCall?
conv5_3/StatefulPartitionedCallStatefulPartitionedCall(conv5_2/StatefulPartitionedCall:output:0conv5_3_1263conv5_3_1265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_3_layer_call_and_return_conditional_losses_7562!
conv5_3/StatefulPartitionedCall?
pool5/PartitionedCallPartitionedCall(conv5_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool5_layer_call_and_return_conditional_losses_5112
pool5/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_5242*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0 ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv2_1/StatefulPartitionedCall ^conv2_2/StatefulPartitionedCall ^conv3_1/StatefulPartitionedCall ^conv3_2/StatefulPartitionedCall ^conv3_3/StatefulPartitionedCall ^conv4_1/StatefulPartitionedCall ^conv4_2/StatefulPartitionedCall ^conv4_3/StatefulPartitionedCall ^conv5_1/StatefulPartitionedCall ^conv5_2/StatefulPartitionedCall ^conv5_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2B
conv2_2/StatefulPartitionedCallconv2_2/StatefulPartitionedCall2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2B
conv4_1/StatefulPartitionedCallconv4_1/StatefulPartitionedCall2B
conv4_2/StatefulPartitionedCallconv4_2/StatefulPartitionedCall2B
conv4_3/StatefulPartitionedCallconv4_3/StatefulPartitionedCall2B
conv5_1/StatefulPartitionedCallconv5_1/StatefulPartitionedCall2B
conv5_2/StatefulPartitionedCallconv5_2/StatefulPartitionedCall2B
conv5_3/StatefulPartitionedCallconv5_3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
A__inference_conv5_3_layer_call_and_return_conditional_losses_1983

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_conv3_3_layer_call_and_return_conditional_losses_652

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
Z
>__inference_pool4_layer_call_and_return_conditional_losses_499

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_vggface_vgg16_layer_call_fn_1519

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_10842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_conv2_1_layer_call_and_return_conditional_losses_1783

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
?
@__inference_conv2_1_layer_call_and_return_conditional_losses_583

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
?
&__inference_conv3_1_layer_call_fn_1812

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_1_layer_call_and_return_conditional_losses_6182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?n
?
 __inference__traced_restore_2172
file_prefix9
assignvariableop_conv1_1_kernel:@-
assignvariableop_1_conv1_1_bias:@;
!assignvariableop_2_conv1_2_kernel:@@-
assignvariableop_3_conv1_2_bias:@<
!assignvariableop_4_conv2_1_kernel:@?.
assignvariableop_5_conv2_1_bias:	?=
!assignvariableop_6_conv2_2_kernel:??.
assignvariableop_7_conv2_2_bias:	?=
!assignvariableop_8_conv3_1_kernel:??.
assignvariableop_9_conv3_1_bias:	?>
"assignvariableop_10_conv3_2_kernel:??/
 assignvariableop_11_conv3_2_bias:	?>
"assignvariableop_12_conv3_3_kernel:??/
 assignvariableop_13_conv3_3_bias:	?>
"assignvariableop_14_conv4_1_kernel:??/
 assignvariableop_15_conv4_1_bias:	?>
"assignvariableop_16_conv4_2_kernel:??/
 assignvariableop_17_conv4_2_bias:	?>
"assignvariableop_18_conv4_3_kernel:??/
 assignvariableop_19_conv4_3_bias:	?>
"assignvariableop_20_conv5_1_kernel:??/
 assignvariableop_21_conv5_1_bias:	?>
"assignvariableop_22_conv5_2_kernel:??/
 assignvariableop_23_conv5_2_bias:	?>
"assignvariableop_24_conv5_3_kernel:??/
 assignvariableop_25_conv5_3_bias:	?
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_conv1_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv2_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv3_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv3_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv3_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_conv3_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv3_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_conv3_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv4_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_conv4_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv4_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_conv4_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv4_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp assignvariableop_19_conv4_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_conv5_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp assignvariableop_21_conv5_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_conv5_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_conv5_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_conv5_3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp assignvariableop_25_conv5_3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26?
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
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
?
?
@__inference_conv5_2_layer_call_and_return_conditional_losses_739

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_conv3_2_layer_call_fn_1832

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_2_layer_call_and_return_conditional_losses_6352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
&__inference_conv4_1_layer_call_fn_1872

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_1_layer_call_and_return_conditional_losses_6702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_conv4_1_layer_call_and_return_conditional_losses_1883

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
m
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_524

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_1405
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_4572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
@__inference_conv1_1_layer_call_and_return_conditional_losses_548

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_conv1_1_layer_call_fn_1732

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_1_layer_call_and_return_conditional_losses_5482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
#__inference_pool2_layer_call_fn_481

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_4752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_conv3_1_layer_call_and_return_conditional_losses_618

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
@__inference_conv4_2_layer_call_and_return_conditional_losses_687

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_pool5_layer_call_fn_517

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool5_layer_call_and_return_conditional_losses_5112
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv3_2_layer_call_and_return_conditional_losses_1843

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
A__inference_conv3_1_layer_call_and_return_conditional_losses_1823

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
@__inference_conv1_2_layer_call_and_return_conditional_losses_565

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
A__inference_conv5_2_layer_call_and_return_conditional_losses_1963

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_vggface_vgg16_layer_call_fn_1196
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_10842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
&__inference_conv2_2_layer_call_fn_1792

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_2_layer_call_and_return_conditional_losses_6002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?
&__inference_conv1_2_layer_call_fn_1752

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_2_layer_call_and_return_conditional_losses_5652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
R
6__inference_global_average_pooling2d_layer_call_fn_530

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_5242
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_conv5_1_layer_call_fn_1932

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_1_layer_call_and_return_conditional_losses_7222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_conv4_2_layer_call_fn_1892

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_2_layer_call_and_return_conditional_losses_6872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_vggface_vgg16_layer_call_fn_820
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_vggface_vgg16_layer_call_and_return_conditional_losses_7652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
??
?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1723

inputs@
&conv1_1_conv2d_readvariableop_resource:@5
'conv1_1_biasadd_readvariableop_resource:@@
&conv1_2_conv2d_readvariableop_resource:@@5
'conv1_2_biasadd_readvariableop_resource:@A
&conv2_1_conv2d_readvariableop_resource:@?6
'conv2_1_biasadd_readvariableop_resource:	?B
&conv2_2_conv2d_readvariableop_resource:??6
'conv2_2_biasadd_readvariableop_resource:	?B
&conv3_1_conv2d_readvariableop_resource:??6
'conv3_1_biasadd_readvariableop_resource:	?B
&conv3_2_conv2d_readvariableop_resource:??6
'conv3_2_biasadd_readvariableop_resource:	?B
&conv3_3_conv2d_readvariableop_resource:??6
'conv3_3_biasadd_readvariableop_resource:	?B
&conv4_1_conv2d_readvariableop_resource:??6
'conv4_1_biasadd_readvariableop_resource:	?B
&conv4_2_conv2d_readvariableop_resource:??6
'conv4_2_biasadd_readvariableop_resource:	?B
&conv4_3_conv2d_readvariableop_resource:??6
'conv4_3_biasadd_readvariableop_resource:	?B
&conv5_1_conv2d_readvariableop_resource:??6
'conv5_1_biasadd_readvariableop_resource:	?B
&conv5_2_conv2d_readvariableop_resource:??6
'conv5_2_biasadd_readvariableop_resource:	?B
&conv5_3_conv2d_readvariableop_resource:??6
'conv5_3_biasadd_readvariableop_resource:	?
identity??conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?conv1_2/BiasAdd/ReadVariableOp?conv1_2/Conv2D/ReadVariableOp?conv2_1/BiasAdd/ReadVariableOp?conv2_1/Conv2D/ReadVariableOp?conv2_2/BiasAdd/ReadVariableOp?conv2_2/Conv2D/ReadVariableOp?conv3_1/BiasAdd/ReadVariableOp?conv3_1/Conv2D/ReadVariableOp?conv3_2/BiasAdd/ReadVariableOp?conv3_2/Conv2D/ReadVariableOp?conv3_3/BiasAdd/ReadVariableOp?conv3_3/Conv2D/ReadVariableOp?conv4_1/BiasAdd/ReadVariableOp?conv4_1/Conv2D/ReadVariableOp?conv4_2/BiasAdd/ReadVariableOp?conv4_2/Conv2D/ReadVariableOp?conv4_3/BiasAdd/ReadVariableOp?conv4_3/Conv2D/ReadVariableOp?conv5_1/BiasAdd/ReadVariableOp?conv5_1/Conv2D/ReadVariableOp?conv5_2/BiasAdd/ReadVariableOp?conv5_2/Conv2D/ReadVariableOp?conv5_3/BiasAdd/ReadVariableOp?conv5_3/Conv2D/ReadVariableOp?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1_1/Conv2D/ReadVariableOp?
conv1_1/Conv2DConv2Dinputs%conv1_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv1_1/Conv2D?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv1_1/BiasAdd/ReadVariableOp?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv1_1/BiasAddz
conv1_1/ReluReluconv1_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv1_1/Relu?
conv1_2/Conv2D/ReadVariableOpReadVariableOp&conv1_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv1_2/Conv2D/ReadVariableOp?
conv1_2/Conv2DConv2Dconv1_1/Relu:activations:0%conv1_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv1_2/Conv2D?
conv1_2/BiasAdd/ReadVariableOpReadVariableOp'conv1_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv1_2/BiasAdd/ReadVariableOp?
conv1_2/BiasAddBiasAddconv1_2/Conv2D:output:0&conv1_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv1_2/BiasAddz
conv1_2/ReluReluconv1_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv1_2/Relu?
pool1/MaxPoolMaxPoolconv1_2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
pool1/MaxPool?
conv2_1/Conv2D/ReadVariableOpReadVariableOp&conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv2_1/Conv2D/ReadVariableOp?
conv2_1/Conv2DConv2Dpool1/MaxPool:output:0%conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
conv2_1/Conv2D?
conv2_1/BiasAdd/ReadVariableOpReadVariableOp'conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv2_1/BiasAdd/ReadVariableOp?
conv2_1/BiasAddBiasAddconv2_1/Conv2D:output:0&conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
conv2_1/BiasAddy
conv2_1/ReluReluconv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
conv2_1/Relu?
conv2_2/Conv2D/ReadVariableOpReadVariableOp&conv2_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2_2/Conv2D/ReadVariableOp?
conv2_2/Conv2DConv2Dconv2_1/Relu:activations:0%conv2_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
conv2_2/Conv2D?
conv2_2/BiasAdd/ReadVariableOpReadVariableOp'conv2_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv2_2/BiasAdd/ReadVariableOp?
conv2_2/BiasAddBiasAddconv2_2/Conv2D:output:0&conv2_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
conv2_2/BiasAddy
conv2_2/ReluReluconv2_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
conv2_2/Relu?
pool2/MaxPoolMaxPoolconv2_2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
pool2/MaxPool?
conv3_1/Conv2D/ReadVariableOpReadVariableOp&conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_1/Conv2D/ReadVariableOp?
conv3_1/Conv2DConv2Dpool2/MaxPool:output:0%conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
conv3_1/Conv2D?
conv3_1/BiasAdd/ReadVariableOpReadVariableOp'conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_1/BiasAdd/ReadVariableOp?
conv3_1/BiasAddBiasAddconv3_1/Conv2D:output:0&conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
conv3_1/BiasAddy
conv3_1/ReluReluconv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
conv3_1/Relu?
conv3_2/Conv2D/ReadVariableOpReadVariableOp&conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_2/Conv2D/ReadVariableOp?
conv3_2/Conv2DConv2Dconv3_1/Relu:activations:0%conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
conv3_2/Conv2D?
conv3_2/BiasAdd/ReadVariableOpReadVariableOp'conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_2/BiasAdd/ReadVariableOp?
conv3_2/BiasAddBiasAddconv3_2/Conv2D:output:0&conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
conv3_2/BiasAddy
conv3_2/ReluReluconv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
conv3_2/Relu?
conv3_3/Conv2D/ReadVariableOpReadVariableOp&conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_3/Conv2D/ReadVariableOp?
conv3_3/Conv2DConv2Dconv3_2/Relu:activations:0%conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
conv3_3/Conv2D?
conv3_3/BiasAdd/ReadVariableOpReadVariableOp'conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_3/BiasAdd/ReadVariableOp?
conv3_3/BiasAddBiasAddconv3_3/Conv2D:output:0&conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
conv3_3/BiasAddy
conv3_3/ReluReluconv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
conv3_3/Relu?
pool3/MaxPoolMaxPoolconv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
pool3/MaxPool?
conv4_1/Conv2D/ReadVariableOpReadVariableOp&conv4_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_1/Conv2D/ReadVariableOp?
conv4_1/Conv2DConv2Dpool3/MaxPool:output:0%conv4_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_1/Conv2D?
conv4_1/BiasAdd/ReadVariableOpReadVariableOp'conv4_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_1/BiasAdd/ReadVariableOp?
conv4_1/BiasAddBiasAddconv4_1/Conv2D:output:0&conv4_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_1/BiasAddy
conv4_1/ReluReluconv4_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_1/Relu?
conv4_2/Conv2D/ReadVariableOpReadVariableOp&conv4_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_2/Conv2D/ReadVariableOp?
conv4_2/Conv2DConv2Dconv4_1/Relu:activations:0%conv4_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_2/Conv2D?
conv4_2/BiasAdd/ReadVariableOpReadVariableOp'conv4_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_2/BiasAdd/ReadVariableOp?
conv4_2/BiasAddBiasAddconv4_2/Conv2D:output:0&conv4_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_2/BiasAddy
conv4_2/ReluReluconv4_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_2/Relu?
conv4_3/Conv2D/ReadVariableOpReadVariableOp&conv4_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_3/Conv2D/ReadVariableOp?
conv4_3/Conv2DConv2Dconv4_2/Relu:activations:0%conv4_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_3/Conv2D?
conv4_3/BiasAdd/ReadVariableOpReadVariableOp'conv4_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_3/BiasAdd/ReadVariableOp?
conv4_3/BiasAddBiasAddconv4_3/Conv2D:output:0&conv4_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_3/BiasAddy
conv4_3/ReluReluconv4_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_3/Relu?
pool4/MaxPoolMaxPoolconv4_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
pool4/MaxPool?
conv5_1/Conv2D/ReadVariableOpReadVariableOp&conv5_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5_1/Conv2D/ReadVariableOp?
conv5_1/Conv2DConv2Dpool4/MaxPool:output:0%conv5_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv5_1/Conv2D?
conv5_1/BiasAdd/ReadVariableOpReadVariableOp'conv5_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv5_1/BiasAdd/ReadVariableOp?
conv5_1/BiasAddBiasAddconv5_1/Conv2D:output:0&conv5_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5_1/BiasAddy
conv5_1/ReluReluconv5_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv5_1/Relu?
conv5_2/Conv2D/ReadVariableOpReadVariableOp&conv5_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5_2/Conv2D/ReadVariableOp?
conv5_2/Conv2DConv2Dconv5_1/Relu:activations:0%conv5_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv5_2/Conv2D?
conv5_2/BiasAdd/ReadVariableOpReadVariableOp'conv5_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv5_2/BiasAdd/ReadVariableOp?
conv5_2/BiasAddBiasAddconv5_2/Conv2D:output:0&conv5_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5_2/BiasAddy
conv5_2/ReluReluconv5_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv5_2/Relu?
conv5_3/Conv2D/ReadVariableOpReadVariableOp&conv5_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5_3/Conv2D/ReadVariableOp?
conv5_3/Conv2DConv2Dconv5_2/Relu:activations:0%conv5_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv5_3/Conv2D?
conv5_3/BiasAdd/ReadVariableOpReadVariableOp'conv5_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv5_3/BiasAdd/ReadVariableOp?
conv5_3/BiasAddBiasAddconv5_3/Conv2D:output:0&conv5_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5_3/BiasAddy
conv5_3/ReluReluconv5_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv5_3/Relu?
pool5/MaxPoolMaxPoolconv5_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
pool5/MaxPool?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeanpool5/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling2d/Mean?
IdentityIdentity&global_average_pooling2d/Mean:output:0^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp^conv1_2/BiasAdd/ReadVariableOp^conv1_2/Conv2D/ReadVariableOp^conv2_1/BiasAdd/ReadVariableOp^conv2_1/Conv2D/ReadVariableOp^conv2_2/BiasAdd/ReadVariableOp^conv2_2/Conv2D/ReadVariableOp^conv3_1/BiasAdd/ReadVariableOp^conv3_1/Conv2D/ReadVariableOp^conv3_2/BiasAdd/ReadVariableOp^conv3_2/Conv2D/ReadVariableOp^conv3_3/BiasAdd/ReadVariableOp^conv3_3/Conv2D/ReadVariableOp^conv4_1/BiasAdd/ReadVariableOp^conv4_1/Conv2D/ReadVariableOp^conv4_2/BiasAdd/ReadVariableOp^conv4_2/Conv2D/ReadVariableOp^conv4_3/BiasAdd/ReadVariableOp^conv4_3/Conv2D/ReadVariableOp^conv5_1/BiasAdd/ReadVariableOp^conv5_1/Conv2D/ReadVariableOp^conv5_2/BiasAdd/ReadVariableOp^conv5_2/Conv2D/ReadVariableOp^conv5_3/BiasAdd/ReadVariableOp^conv5_3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2@
conv1_2/BiasAdd/ReadVariableOpconv1_2/BiasAdd/ReadVariableOp2>
conv1_2/Conv2D/ReadVariableOpconv1_2/Conv2D/ReadVariableOp2@
conv2_1/BiasAdd/ReadVariableOpconv2_1/BiasAdd/ReadVariableOp2>
conv2_1/Conv2D/ReadVariableOpconv2_1/Conv2D/ReadVariableOp2@
conv2_2/BiasAdd/ReadVariableOpconv2_2/BiasAdd/ReadVariableOp2>
conv2_2/Conv2D/ReadVariableOpconv2_2/Conv2D/ReadVariableOp2@
conv3_1/BiasAdd/ReadVariableOpconv3_1/BiasAdd/ReadVariableOp2>
conv3_1/Conv2D/ReadVariableOpconv3_1/Conv2D/ReadVariableOp2@
conv3_2/BiasAdd/ReadVariableOpconv3_2/BiasAdd/ReadVariableOp2>
conv3_2/Conv2D/ReadVariableOpconv3_2/Conv2D/ReadVariableOp2@
conv3_3/BiasAdd/ReadVariableOpconv3_3/BiasAdd/ReadVariableOp2>
conv3_3/Conv2D/ReadVariableOpconv3_3/Conv2D/ReadVariableOp2@
conv4_1/BiasAdd/ReadVariableOpconv4_1/BiasAdd/ReadVariableOp2>
conv4_1/Conv2D/ReadVariableOpconv4_1/Conv2D/ReadVariableOp2@
conv4_2/BiasAdd/ReadVariableOpconv4_2/BiasAdd/ReadVariableOp2>
conv4_2/Conv2D/ReadVariableOpconv4_2/Conv2D/ReadVariableOp2@
conv4_3/BiasAdd/ReadVariableOpconv4_3/BiasAdd/ReadVariableOp2>
conv4_3/Conv2D/ReadVariableOpconv4_3/Conv2D/ReadVariableOp2@
conv5_1/BiasAdd/ReadVariableOpconv5_1/BiasAdd/ReadVariableOp2>
conv5_1/Conv2D/ReadVariableOpconv5_1/Conv2D/ReadVariableOp2@
conv5_2/BiasAdd/ReadVariableOpconv5_2/BiasAdd/ReadVariableOp2>
conv5_2/Conv2D/ReadVariableOpconv5_2/Conv2D/ReadVariableOp2@
conv5_3/BiasAdd/ReadVariableOpconv5_3/BiasAdd/ReadVariableOp2>
conv5_3/Conv2D/ReadVariableOpconv5_3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
#__inference_pool4_layer_call_fn_505

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool4_layer_call_and_return_conditional_losses_4992
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_conv2_1_layer_call_fn_1772

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_1_layer_call_and_return_conditional_losses_5832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?;
?

__inference__traced_save_2084
file_prefix-
)savev2_conv1_1_kernel_read_readvariableop+
'savev2_conv1_1_bias_read_readvariableop-
)savev2_conv1_2_kernel_read_readvariableop+
'savev2_conv1_2_bias_read_readvariableop-
)savev2_conv2_1_kernel_read_readvariableop+
'savev2_conv2_1_bias_read_readvariableop-
)savev2_conv2_2_kernel_read_readvariableop+
'savev2_conv2_2_bias_read_readvariableop-
)savev2_conv3_1_kernel_read_readvariableop+
'savev2_conv3_1_bias_read_readvariableop-
)savev2_conv3_2_kernel_read_readvariableop+
'savev2_conv3_2_bias_read_readvariableop-
)savev2_conv3_3_kernel_read_readvariableop+
'savev2_conv3_3_bias_read_readvariableop-
)savev2_conv4_1_kernel_read_readvariableop+
'savev2_conv4_1_bias_read_readvariableop-
)savev2_conv4_2_kernel_read_readvariableop+
'savev2_conv4_2_bias_read_readvariableop-
)savev2_conv4_3_kernel_read_readvariableop+
'savev2_conv4_3_bias_read_readvariableop-
)savev2_conv5_1_kernel_read_readvariableop+
'savev2_conv5_1_bias_read_readvariableop-
)savev2_conv5_2_kernel_read_readvariableop+
'savev2_conv5_2_bias_read_readvariableop-
)savev2_conv5_3_kernel_read_readvariableop+
'savev2_conv5_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_conv1_1_kernel_read_readvariableop'savev2_conv1_1_bias_read_readvariableop)savev2_conv1_2_kernel_read_readvariableop'savev2_conv1_2_bias_read_readvariableop)savev2_conv2_1_kernel_read_readvariableop'savev2_conv2_1_bias_read_readvariableop)savev2_conv2_2_kernel_read_readvariableop'savev2_conv2_2_bias_read_readvariableop)savev2_conv3_1_kernel_read_readvariableop'savev2_conv3_1_bias_read_readvariableop)savev2_conv3_2_kernel_read_readvariableop'savev2_conv3_2_bias_read_readvariableop)savev2_conv3_3_kernel_read_readvariableop'savev2_conv3_3_bias_read_readvariableop)savev2_conv4_1_kernel_read_readvariableop'savev2_conv4_1_bias_read_readvariableop)savev2_conv4_2_kernel_read_readvariableop'savev2_conv4_2_bias_read_readvariableop)savev2_conv4_3_kernel_read_readvariableop'savev2_conv4_3_bias_read_readvariableop)savev2_conv5_1_kernel_read_readvariableop'savev2_conv5_1_bias_read_readvariableop)savev2_conv5_2_kernel_read_readvariableop'savev2_conv5_2_bias_read_readvariableop)savev2_conv5_3_kernel_read_readvariableop'savev2_conv5_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.	*
(
_output_shapes
:??:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:

_output_shapes
: 
?
?
&__inference_conv5_2_layer_call_fn_1952

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_2_layer_call_and_return_conditional_losses_7392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_conv5_1_layer_call_and_return_conditional_losses_1943

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_conv4_3_layer_call_fn_1912

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_3_layer_call_and_return_conditional_losses_7042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?V
?
F__inference_vggface_vgg16_layer_call_and_return_conditional_losses_765

inputs%
conv1_1_549:@
conv1_1_551:@%
conv1_2_566:@@
conv1_2_568:@&
conv2_1_584:@?
conv2_1_586:	?'
conv2_2_601:??
conv2_2_603:	?'
conv3_1_619:??
conv3_1_621:	?'
conv3_2_636:??
conv3_2_638:	?'
conv3_3_653:??
conv3_3_655:	?'
conv4_1_671:??
conv4_1_673:	?'
conv4_2_688:??
conv4_2_690:	?'
conv4_3_705:??
conv4_3_707:	?'
conv5_1_723:??
conv5_1_725:	?'
conv5_2_740:??
conv5_2_742:	?'
conv5_3_757:??
conv5_3_759:	?
identity??conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv2_1/StatefulPartitionedCall?conv2_2/StatefulPartitionedCall?conv3_1/StatefulPartitionedCall?conv3_2/StatefulPartitionedCall?conv3_3/StatefulPartitionedCall?conv4_1/StatefulPartitionedCall?conv4_2/StatefulPartitionedCall?conv4_3/StatefulPartitionedCall?conv5_1/StatefulPartitionedCall?conv5_2/StatefulPartitionedCall?conv5_3/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_1_549conv1_1_551*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_1_layer_call_and_return_conditional_losses_5482!
conv1_1/StatefulPartitionedCall?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0conv1_2_566conv1_2_568*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_2_layer_call_and_return_conditional_losses_5652!
conv1_2/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_4632
pool1/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_1_584conv2_1_586*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_1_layer_call_and_return_conditional_losses_5832!
conv2_1/StatefulPartitionedCall?
conv2_2/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0conv2_2_601conv2_2_603*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_2_layer_call_and_return_conditional_losses_6002!
conv2_2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall(conv2_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_4752
pool2/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_1_619conv3_1_621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_1_layer_call_and_return_conditional_losses_6182!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_636conv3_2_638*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_2_layer_call_and_return_conditional_losses_6352!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_653conv3_3_655*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_3_layer_call_and_return_conditional_losses_6522!
conv3_3/StatefulPartitionedCall?
pool3/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_4872
pool3/PartitionedCall?
conv4_1/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_1_671conv4_1_673*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_1_layer_call_and_return_conditional_losses_6702!
conv4_1/StatefulPartitionedCall?
conv4_2/StatefulPartitionedCallStatefulPartitionedCall(conv4_1/StatefulPartitionedCall:output:0conv4_2_688conv4_2_690*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_2_layer_call_and_return_conditional_losses_6872!
conv4_2/StatefulPartitionedCall?
conv4_3/StatefulPartitionedCallStatefulPartitionedCall(conv4_2/StatefulPartitionedCall:output:0conv4_3_705conv4_3_707*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_3_layer_call_and_return_conditional_losses_7042!
conv4_3/StatefulPartitionedCall?
pool4/PartitionedCallPartitionedCall(conv4_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool4_layer_call_and_return_conditional_losses_4992
pool4/PartitionedCall?
conv5_1/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0conv5_1_723conv5_1_725*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_1_layer_call_and_return_conditional_losses_7222!
conv5_1/StatefulPartitionedCall?
conv5_2/StatefulPartitionedCallStatefulPartitionedCall(conv5_1/StatefulPartitionedCall:output:0conv5_2_740conv5_2_742*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_2_layer_call_and_return_conditional_losses_7392!
conv5_2/StatefulPartitionedCall?
conv5_3/StatefulPartitionedCallStatefulPartitionedCall(conv5_2/StatefulPartitionedCall:output:0conv5_3_757conv5_3_759*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_3_layer_call_and_return_conditional_losses_7562!
conv5_3/StatefulPartitionedCall?
pool5/PartitionedCallPartitionedCall(conv5_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool5_layer_call_and_return_conditional_losses_5112
pool5/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_5242*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0 ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv2_1/StatefulPartitionedCall ^conv2_2/StatefulPartitionedCall ^conv3_1/StatefulPartitionedCall ^conv3_2/StatefulPartitionedCall ^conv3_3/StatefulPartitionedCall ^conv4_1/StatefulPartitionedCall ^conv4_2/StatefulPartitionedCall ^conv4_3/StatefulPartitionedCall ^conv5_1/StatefulPartitionedCall ^conv5_2/StatefulPartitionedCall ^conv5_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2B
conv2_2/StatefulPartitionedCallconv2_2/StatefulPartitionedCall2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2B
conv4_1/StatefulPartitionedCallconv4_1/StatefulPartitionedCall2B
conv4_2/StatefulPartitionedCallconv4_2/StatefulPartitionedCall2B
conv4_3/StatefulPartitionedCallconv4_3/StatefulPartitionedCall2B
conv5_1/StatefulPartitionedCallconv5_1/StatefulPartitionedCall2B
conv5_2/StatefulPartitionedCallconv5_2/StatefulPartitionedCall2B
conv5_3/StatefulPartitionedCallconv5_3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
#__inference_pool1_layer_call_fn_469

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_4632
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv2_2_layer_call_and_return_conditional_losses_1803

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?V
?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1084

inputs&
conv1_1_1012:@
conv1_1_1014:@&
conv1_2_1017:@@
conv1_2_1019:@'
conv2_1_1023:@?
conv2_1_1025:	?(
conv2_2_1028:??
conv2_2_1030:	?(
conv3_1_1034:??
conv3_1_1036:	?(
conv3_2_1039:??
conv3_2_1041:	?(
conv3_3_1044:??
conv3_3_1046:	?(
conv4_1_1050:??
conv4_1_1052:	?(
conv4_2_1055:??
conv4_2_1057:	?(
conv4_3_1060:??
conv4_3_1062:	?(
conv5_1_1066:??
conv5_1_1068:	?(
conv5_2_1071:??
conv5_2_1073:	?(
conv5_3_1076:??
conv5_3_1078:	?
identity??conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv2_1/StatefulPartitionedCall?conv2_2/StatefulPartitionedCall?conv3_1/StatefulPartitionedCall?conv3_2/StatefulPartitionedCall?conv3_3/StatefulPartitionedCall?conv4_1/StatefulPartitionedCall?conv4_2/StatefulPartitionedCall?conv4_3/StatefulPartitionedCall?conv5_1/StatefulPartitionedCall?conv5_2/StatefulPartitionedCall?conv5_3/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_1_1012conv1_1_1014*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_1_layer_call_and_return_conditional_losses_5482!
conv1_1/StatefulPartitionedCall?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0conv1_2_1017conv1_2_1019*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_2_layer_call_and_return_conditional_losses_5652!
conv1_2/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_4632
pool1/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_1_1023conv2_1_1025*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_1_layer_call_and_return_conditional_losses_5832!
conv2_1/StatefulPartitionedCall?
conv2_2/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0conv2_2_1028conv2_2_1030*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_2_layer_call_and_return_conditional_losses_6002!
conv2_2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall(conv2_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_4752
pool2/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_1_1034conv3_1_1036*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_1_layer_call_and_return_conditional_losses_6182!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_1039conv3_2_1041*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_2_layer_call_and_return_conditional_losses_6352!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_1044conv3_3_1046*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_3_layer_call_and_return_conditional_losses_6522!
conv3_3/StatefulPartitionedCall?
pool3/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_4872
pool3/PartitionedCall?
conv4_1/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_1_1050conv4_1_1052*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_1_layer_call_and_return_conditional_losses_6702!
conv4_1/StatefulPartitionedCall?
conv4_2/StatefulPartitionedCallStatefulPartitionedCall(conv4_1/StatefulPartitionedCall:output:0conv4_2_1055conv4_2_1057*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_2_layer_call_and_return_conditional_losses_6872!
conv4_2/StatefulPartitionedCall?
conv4_3/StatefulPartitionedCallStatefulPartitionedCall(conv4_2/StatefulPartitionedCall:output:0conv4_3_1060conv4_3_1062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_3_layer_call_and_return_conditional_losses_7042!
conv4_3/StatefulPartitionedCall?
pool4/PartitionedCallPartitionedCall(conv4_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool4_layer_call_and_return_conditional_losses_4992
pool4/PartitionedCall?
conv5_1/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0conv5_1_1066conv5_1_1068*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_1_layer_call_and_return_conditional_losses_7222!
conv5_1/StatefulPartitionedCall?
conv5_2/StatefulPartitionedCallStatefulPartitionedCall(conv5_1/StatefulPartitionedCall:output:0conv5_2_1071conv5_2_1073*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_2_layer_call_and_return_conditional_losses_7392!
conv5_2/StatefulPartitionedCall?
conv5_3/StatefulPartitionedCallStatefulPartitionedCall(conv5_2/StatefulPartitionedCall:output:0conv5_3_1076conv5_3_1078*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_3_layer_call_and_return_conditional_losses_7562!
conv5_3/StatefulPartitionedCall?
pool5/PartitionedCallPartitionedCall(conv5_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool5_layer_call_and_return_conditional_losses_5112
pool5/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_5242*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0 ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv2_1/StatefulPartitionedCall ^conv2_2/StatefulPartitionedCall ^conv3_1/StatefulPartitionedCall ^conv3_2/StatefulPartitionedCall ^conv3_3/StatefulPartitionedCall ^conv4_1/StatefulPartitionedCall ^conv4_2/StatefulPartitionedCall ^conv4_3/StatefulPartitionedCall ^conv5_1/StatefulPartitionedCall ^conv5_2/StatefulPartitionedCall ^conv5_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2B
conv2_2/StatefulPartitionedCallconv2_2/StatefulPartitionedCall2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2B
conv4_1/StatefulPartitionedCallconv4_1/StatefulPartitionedCall2B
conv4_2/StatefulPartitionedCallconv4_2/StatefulPartitionedCall2B
conv4_3/StatefulPartitionedCallconv4_3/StatefulPartitionedCall2B
conv5_1/StatefulPartitionedCallconv5_1/StatefulPartitionedCall2B
conv5_2/StatefulPartitionedCallconv5_2/StatefulPartitionedCall2B
conv5_3/StatefulPartitionedCallconv5_3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_conv1_2_layer_call_and_return_conditional_losses_1763

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
,__inference_vggface_vgg16_layer_call_fn_1462

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_vggface_vgg16_layer_call_and_return_conditional_losses_7652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Z
>__inference_pool5_layer_call_and_return_conditional_losses_511

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?V
?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1346
input_1&
conv1_1_1274:@
conv1_1_1276:@&
conv1_2_1279:@@
conv1_2_1281:@'
conv2_1_1285:@?
conv2_1_1287:	?(
conv2_2_1290:??
conv2_2_1292:	?(
conv3_1_1296:??
conv3_1_1298:	?(
conv3_2_1301:??
conv3_2_1303:	?(
conv3_3_1306:??
conv3_3_1308:	?(
conv4_1_1312:??
conv4_1_1314:	?(
conv4_2_1317:??
conv4_2_1319:	?(
conv4_3_1322:??
conv4_3_1324:	?(
conv5_1_1328:??
conv5_1_1330:	?(
conv5_2_1333:??
conv5_2_1335:	?(
conv5_3_1338:??
conv5_3_1340:	?
identity??conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv2_1/StatefulPartitionedCall?conv2_2/StatefulPartitionedCall?conv3_1/StatefulPartitionedCall?conv3_2/StatefulPartitionedCall?conv3_3/StatefulPartitionedCall?conv4_1/StatefulPartitionedCall?conv4_2/StatefulPartitionedCall?conv4_3/StatefulPartitionedCall?conv5_1/StatefulPartitionedCall?conv5_2/StatefulPartitionedCall?conv5_3/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_1_1274conv1_1_1276*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_1_layer_call_and_return_conditional_losses_5482!
conv1_1/StatefulPartitionedCall?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0conv1_2_1279conv1_2_1281*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_2_layer_call_and_return_conditional_losses_5652!
conv1_2/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_4632
pool1/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_1_1285conv2_1_1287*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_1_layer_call_and_return_conditional_losses_5832!
conv2_1/StatefulPartitionedCall?
conv2_2/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0conv2_2_1290conv2_2_1292*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2_2_layer_call_and_return_conditional_losses_6002!
conv2_2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall(conv2_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_4752
pool2/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_1_1296conv3_1_1298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_1_layer_call_and_return_conditional_losses_6182!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_1301conv3_2_1303*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_2_layer_call_and_return_conditional_losses_6352!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_1306conv3_3_1308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv3_3_layer_call_and_return_conditional_losses_6522!
conv3_3/StatefulPartitionedCall?
pool3/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_4872
pool3/PartitionedCall?
conv4_1/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_1_1312conv4_1_1314*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_1_layer_call_and_return_conditional_losses_6702!
conv4_1/StatefulPartitionedCall?
conv4_2/StatefulPartitionedCallStatefulPartitionedCall(conv4_1/StatefulPartitionedCall:output:0conv4_2_1317conv4_2_1319*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_2_layer_call_and_return_conditional_losses_6872!
conv4_2/StatefulPartitionedCall?
conv4_3/StatefulPartitionedCallStatefulPartitionedCall(conv4_2/StatefulPartitionedCall:output:0conv4_3_1322conv4_3_1324*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv4_3_layer_call_and_return_conditional_losses_7042!
conv4_3/StatefulPartitionedCall?
pool4/PartitionedCallPartitionedCall(conv4_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool4_layer_call_and_return_conditional_losses_4992
pool4/PartitionedCall?
conv5_1/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0conv5_1_1328conv5_1_1330*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_1_layer_call_and_return_conditional_losses_7222!
conv5_1/StatefulPartitionedCall?
conv5_2/StatefulPartitionedCallStatefulPartitionedCall(conv5_1/StatefulPartitionedCall:output:0conv5_2_1333conv5_2_1335*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_2_layer_call_and_return_conditional_losses_7392!
conv5_2/StatefulPartitionedCall?
conv5_3/StatefulPartitionedCallStatefulPartitionedCall(conv5_2/StatefulPartitionedCall:output:0conv5_3_1338conv5_3_1340*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv5_3_layer_call_and_return_conditional_losses_7562!
conv5_3/StatefulPartitionedCall?
pool5/PartitionedCallPartitionedCall(conv5_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_pool5_layer_call_and_return_conditional_losses_5112
pool5/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_5242*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0 ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv2_1/StatefulPartitionedCall ^conv2_2/StatefulPartitionedCall ^conv3_1/StatefulPartitionedCall ^conv3_2/StatefulPartitionedCall ^conv3_3/StatefulPartitionedCall ^conv4_1/StatefulPartitionedCall ^conv4_2/StatefulPartitionedCall ^conv4_3/StatefulPartitionedCall ^conv5_1/StatefulPartitionedCall ^conv5_2/StatefulPartitionedCall ^conv5_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2B
conv2_2/StatefulPartitionedCallconv2_2/StatefulPartitionedCall2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2B
conv4_1/StatefulPartitionedCallconv4_1/StatefulPartitionedCall2B
conv4_2/StatefulPartitionedCallconv4_2/StatefulPartitionedCall2B
conv4_3/StatefulPartitionedCallconv4_3/StatefulPartitionedCall2B
conv5_1/StatefulPartitionedCallconv5_1/StatefulPartitionedCall2B
conv5_2/StatefulPartitionedCallconv5_2/StatefulPartitionedCall2B
conv5_3/StatefulPartitionedCallconv5_3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
A__inference_conv4_3_layer_call_and_return_conditional_losses_1923

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Z
>__inference_pool1_layer_call_and_return_conditional_losses_463

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv1_1_layer_call_and_return_conditional_losses_1743

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_conv5_3_layer_call_and_return_conditional_losses_756

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????M
global_average_pooling2d1
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer-19
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"name": "vggface_vgg16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "vggface_vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_2", "inbound_nodes": [[["conv1_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1", "inbound_nodes": [[["conv1_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_1", "inbound_nodes": [[["pool1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_2", "inbound_nodes": [[["conv2_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2", "inbound_nodes": [[["conv2_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_1", "inbound_nodes": [[["pool2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_2", "inbound_nodes": [[["conv3_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_3", "inbound_nodes": [[["conv3_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool3", "inbound_nodes": [[["conv3_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_1", "inbound_nodes": [[["pool3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_2", "inbound_nodes": [[["conv4_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_3", "inbound_nodes": [[["conv4_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool4", "inbound_nodes": [[["conv4_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_1", "inbound_nodes": [[["pool4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_2", "inbound_nodes": [[["conv5_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_3", "inbound_nodes": [[["conv5_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool5", "inbound_nodes": [[["conv5_3", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["pool5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["global_average_pooling2d", 0, 0]]}, "shared_object_id": 46, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vggface_vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv1_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_2", "inbound_nodes": [[["conv1_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1", "inbound_nodes": [[["conv1_2", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_1", "inbound_nodes": [[["pool1", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv2_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_2", "inbound_nodes": [[["conv2_1", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "MaxPooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2", "inbound_nodes": [[["conv2_2", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_1", "inbound_nodes": [[["pool2", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_2", "inbound_nodes": [[["conv3_1", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_3", "inbound_nodes": [[["conv3_2", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "MaxPooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool3", "inbound_nodes": [[["conv3_3", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "Conv2D", "config": {"name": "conv4_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_1", "inbound_nodes": [[["pool3", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Conv2D", "config": {"name": "conv4_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_2", "inbound_nodes": [[["conv4_1", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Conv2D", "config": {"name": "conv4_3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_3", "inbound_nodes": [[["conv4_2", 0, 0, {}]]], "shared_object_id": 33}, {"class_name": "MaxPooling2D", "config": {"name": "pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool4", "inbound_nodes": [[["conv4_3", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "Conv2D", "config": {"name": "conv5_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_1", "inbound_nodes": [[["pool4", 0, 0, {}]]], "shared_object_id": 37}, {"class_name": "Conv2D", "config": {"name": "conv5_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_2", "inbound_nodes": [[["conv5_1", 0, 0, {}]]], "shared_object_id": 40}, {"class_name": "Conv2D", "config": {"name": "conv5_3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_3", "inbound_nodes": [[["conv5_2", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "MaxPooling2D", "config": {"name": "pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool5", "inbound_nodes": [[["conv5_3", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["pool5", 0, 0, {}]]], "shared_object_id": 45}], "input_layers": [["input_1", 0, 0]], "output_layers": [["global_average_pooling2d", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv1_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}}
?


 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv1_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv1_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 64]}}
?
&trainable_variables
'regularization_losses
(	variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv1_2", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 50}}
?


*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool1", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 64]}}
?

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 128]}}
?
6trainable_variables
7regularization_losses
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2_2", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 53}}
?


:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv3_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool2", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 128]}}
?

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv3_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv3_1", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 256]}}
?

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv3_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv3_2", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 256]}}
?
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv3_3", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 57}}
?


Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv4_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv4_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool3", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 256]}}
?

Vkernel
Wbias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv4_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv4_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv4_1", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 512]}}
?

\kernel
]bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv4_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv4_3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv4_2", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 512]}}
?
btrainable_variables
cregularization_losses
d	variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv4_3", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 61}}
?


fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv5_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv5_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool4", 0, 0, {}]]], "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv5_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv5_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv5_1", 0, 0, {}]]], "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?

rkernel
sbias
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv5_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv5_3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv5_2", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv5_3", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 65}}
?
|trainable_variables
}regularization_losses
~	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool5", 0, 0, {}]]], "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 66}}
?
0
1
 2
!3
*4
+5
06
17
:8
;9
@10
A11
F12
G13
P14
Q15
V16
W17
\18
]19
f20
g21
l22
m23
r24
s25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
 2
!3
*4
+5
06
17
:8
;9
@10
A11
F12
G13
P14
Q15
V16
W17
\18
]19
f20
g21
l22
m23
r24
s25"
trackable_list_wrapper
?
 ?layer_regularization_losses
trainable_variables
?layers
regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
(:&@2conv1_1/kernel
:@2conv1_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 ?layer_regularization_losses
trainable_variables
?layers
regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&@@2conv1_2/kernel
:@2conv1_2/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
 ?layer_regularization_losses
"trainable_variables
?layers
#regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
$	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
&trainable_variables
?layers
'regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
(	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@?2conv2_1/kernel
:?2conv2_1/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
 ?layer_regularization_losses
,trainable_variables
?layers
-regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
.	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv2_2/kernel
:?2conv2_2/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
 ?layer_regularization_losses
2trainable_variables
?layers
3regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
4	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
6trainable_variables
?layers
7regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
8	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv3_1/kernel
:?2conv3_1/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
 ?layer_regularization_losses
<trainable_variables
?layers
=regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
>	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv3_2/kernel
:?2conv3_2/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Btrainable_variables
?layers
Cregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
D	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv3_3/kernel
:?2conv3_3/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Htrainable_variables
?layers
Iregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
J	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Ltrainable_variables
?layers
Mregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
N	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv4_1/kernel
:?2conv4_1/bias
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Rtrainable_variables
?layers
Sregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
T	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv4_2/kernel
:?2conv4_2/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Xtrainable_variables
?layers
Yregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
Z	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv4_3/kernel
:?2conv4_3/bias
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
?
 ?layer_regularization_losses
^trainable_variables
?layers
_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
`	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
btrainable_variables
?layers
cregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
d	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv5_1/kernel
:?2conv5_1/bias
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
 ?layer_regularization_losses
htrainable_variables
?layers
iregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
j	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv5_2/kernel
:?2conv5_2/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
 ?layer_regularization_losses
ntrainable_variables
?layers
oregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
p	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv5_3/kernel
:?2conv5_3/bias
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
?
 ?layer_regularization_losses
ttrainable_variables
?layers
uregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
v	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
xtrainable_variables
?layers
yregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
z	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
|trainable_variables
?layers
}regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
~	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
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
?2?
__inference__wrapped_model_457?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_1???????????
?2?
+__inference_vggface_vgg16_layer_call_fn_820
,__inference_vggface_vgg16_layer_call_fn_1462
,__inference_vggface_vgg16_layer_call_fn_1519
,__inference_vggface_vgg16_layer_call_fn_1196?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1621
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1723
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1271
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1346?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_conv1_1_layer_call_fn_1732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv1_1_layer_call_and_return_conditional_losses_1743?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv1_2_layer_call_fn_1752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv1_2_layer_call_and_return_conditional_losses_1763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_pool1_layer_call_fn_469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
>__inference_pool1_layer_call_and_return_conditional_losses_463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_conv2_1_layer_call_fn_1772?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2_1_layer_call_and_return_conditional_losses_1783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv2_2_layer_call_fn_1792?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2_2_layer_call_and_return_conditional_losses_1803?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_pool2_layer_call_fn_481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
>__inference_pool2_layer_call_and_return_conditional_losses_475?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_conv3_1_layer_call_fn_1812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv3_1_layer_call_and_return_conditional_losses_1823?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv3_2_layer_call_fn_1832?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv3_2_layer_call_and_return_conditional_losses_1843?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv3_3_layer_call_fn_1852?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv3_3_layer_call_and_return_conditional_losses_1863?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_pool3_layer_call_fn_493?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
>__inference_pool3_layer_call_and_return_conditional_losses_487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_conv4_1_layer_call_fn_1872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv4_1_layer_call_and_return_conditional_losses_1883?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv4_2_layer_call_fn_1892?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv4_2_layer_call_and_return_conditional_losses_1903?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv4_3_layer_call_fn_1912?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv4_3_layer_call_and_return_conditional_losses_1923?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_pool4_layer_call_fn_505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
>__inference_pool4_layer_call_and_return_conditional_losses_499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_conv5_1_layer_call_fn_1932?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv5_1_layer_call_and_return_conditional_losses_1943?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv5_2_layer_call_fn_1952?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv5_2_layer_call_and_return_conditional_losses_1963?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv5_3_layer_call_fn_1972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv5_3_layer_call_and_return_conditional_losses_1983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_pool5_layer_call_fn_517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
>__inference_pool5_layer_call_and_return_conditional_losses_511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
6__inference_global_average_pooling2d_layer_call_fn_530?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
"__inference_signature_wrapper_1405input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_457? !*+01:;@AFGPQVW\]fglmrs:?7
0?-
+?(
input_1???????????
? "T?Q
O
global_average_pooling2d3?0
global_average_pooling2d???????????
A__inference_conv1_1_layer_call_and_return_conditional_losses_1743p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
&__inference_conv1_1_layer_call_fn_1732c9?6
/?,
*?'
inputs???????????
? ""????????????@?
A__inference_conv1_2_layer_call_and_return_conditional_losses_1763p !9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
&__inference_conv1_2_layer_call_fn_1752c !9?6
/?,
*?'
inputs???????????@
? ""????????????@?
A__inference_conv2_1_layer_call_and_return_conditional_losses_1783m*+7?4
-?*
(?%
inputs?????????pp@
? ".?+
$?!
0?????????pp?
? ?
&__inference_conv2_1_layer_call_fn_1772`*+7?4
-?*
(?%
inputs?????????pp@
? "!??????????pp??
A__inference_conv2_2_layer_call_and_return_conditional_losses_1803n018?5
.?+
)?&
inputs?????????pp?
? ".?+
$?!
0?????????pp?
? ?
&__inference_conv2_2_layer_call_fn_1792a018?5
.?+
)?&
inputs?????????pp?
? "!??????????pp??
A__inference_conv3_1_layer_call_and_return_conditional_losses_1823n:;8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
&__inference_conv3_1_layer_call_fn_1812a:;8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
A__inference_conv3_2_layer_call_and_return_conditional_losses_1843n@A8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
&__inference_conv3_2_layer_call_fn_1832a@A8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
A__inference_conv3_3_layer_call_and_return_conditional_losses_1863nFG8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
&__inference_conv3_3_layer_call_fn_1852aFG8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
A__inference_conv4_1_layer_call_and_return_conditional_losses_1883nPQ8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
&__inference_conv4_1_layer_call_fn_1872aPQ8?5
.?+
)?&
inputs??????????
? "!????????????
A__inference_conv4_2_layer_call_and_return_conditional_losses_1903nVW8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
&__inference_conv4_2_layer_call_fn_1892aVW8?5
.?+
)?&
inputs??????????
? "!????????????
A__inference_conv4_3_layer_call_and_return_conditional_losses_1923n\]8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
&__inference_conv4_3_layer_call_fn_1912a\]8?5
.?+
)?&
inputs??????????
? "!????????????
A__inference_conv5_1_layer_call_and_return_conditional_losses_1943nfg8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
&__inference_conv5_1_layer_call_fn_1932afg8?5
.?+
)?&
inputs??????????
? "!????????????
A__inference_conv5_2_layer_call_and_return_conditional_losses_1963nlm8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
&__inference_conv5_2_layer_call_fn_1952alm8?5
.?+
)?&
inputs??????????
? "!????????????
A__inference_conv5_3_layer_call_and_return_conditional_losses_1983nrs8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
&__inference_conv5_3_layer_call_fn_1972ars8?5
.?+
)?&
inputs??????????
? "!????????????
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_524?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
6__inference_global_average_pooling2d_layer_call_fn_530wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
>__inference_pool1_layer_call_and_return_conditional_losses_463?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
#__inference_pool1_layer_call_fn_469?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
>__inference_pool2_layer_call_and_return_conditional_losses_475?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
#__inference_pool2_layer_call_fn_481?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
>__inference_pool3_layer_call_and_return_conditional_losses_487?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
#__inference_pool3_layer_call_fn_493?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
>__inference_pool4_layer_call_and_return_conditional_losses_499?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
#__inference_pool4_layer_call_fn_505?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
>__inference_pool5_layer_call_and_return_conditional_losses_511?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
#__inference_pool5_layer_call_fn_517?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
"__inference_signature_wrapper_1405? !*+01:;@AFGPQVW\]fglmrsE?B
? 
;?8
6
input_1+?(
input_1???????????"T?Q
O
global_average_pooling2d3?0
global_average_pooling2d???????????
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1271? !*+01:;@AFGPQVW\]fglmrsB??
8?5
+?(
input_1???????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1346? !*+01:;@AFGPQVW\]fglmrsB??
8?5
+?(
input_1???????????
p

 
? "&?#
?
0??????????
? ?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1621? !*+01:;@AFGPQVW\]fglmrsA?>
7?4
*?'
inputs???????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_vggface_vgg16_layer_call_and_return_conditional_losses_1723? !*+01:;@AFGPQVW\]fglmrsA?>
7?4
*?'
inputs???????????
p

 
? "&?#
?
0??????????
? ?
,__inference_vggface_vgg16_layer_call_fn_1196{ !*+01:;@AFGPQVW\]fglmrsB??
8?5
+?(
input_1???????????
p

 
? "????????????
,__inference_vggface_vgg16_layer_call_fn_1462z !*+01:;@AFGPQVW\]fglmrsA?>
7?4
*?'
inputs???????????
p 

 
? "????????????
,__inference_vggface_vgg16_layer_call_fn_1519z !*+01:;@AFGPQVW\]fglmrsA?>
7?4
*?'
inputs???????????
p

 
? "????????????
+__inference_vggface_vgg16_layer_call_fn_820{ !*+01:;@AFGPQVW\]fglmrsB??
8?5
+?(
input_1???????????
p 

 
? "???????????