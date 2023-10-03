# in the name of God the most compassionate the most meciful 

# lets create a Vector Class, in two forms, one taking only 2 numbers, representing x,y
# and the other an ndimensional vector. 
# the attributes need to be readonly (i.e. data attributes like x,y)
# the class needs to be immutable, so it cant be changed inplace, and all changes returns a new object 
# it needs to support sequence/iterable behavior
# use slots to conserve memory, write down its shortcommings as well
# to make this immutable, we need to  calculate its unique hash, and make it readonly
# make this support __bytes__ and __format__
import math
from array import array 

class Vector2d:
    # since we want to support bytes representation, we specify the bytetype as well
    typecode ='d'
    
    # since we have bytes, lets add ability to load from bytes, 
    # this would return an instance from existing bytes array, 
    # however, to make it more efficient, we can use memoryview
    # which allows us to access the underlying memory (byte array)
    # without copying anything!
    # 
    
    # so we can code it as a class method, remember to mark it as a class method
    # we must use @classmethod property on it! otherwise we have to provide the class ourselve when calling this method!
    @classmethod
    def from_bytes(cls, byte_array):
        # we can use array and read it back and create a new instance from it 
        # but since we added a typecode in the front of the list we had to remove it first
        # otherwise we face an error, array() as you remember encodes the typecode itself and doesnt need it
        # arr = array(cls.typecode, byte_array[1:])
        # print(arr)
        # return cls(*arr)
        # but theres a better way, we can create a view from the bytesarray in memory and 
        # avoid copying data! 
        # thats why we encoded the datatype in the begining! so if we chose this second way, we can retrieve the dtype!
        # convert and get back the dtype of our array, remember we need character codes like 'd', 'f', etc
        array_dtype = chr(byte_array[0])
        print(f'dtype: {array_dtype}')
        # now lets access the underlying data using memoryview
        # since want the data only, we chose the array data past the [0]s element which was the dtype!
        # and cast the result back to the actual dtupe.
        view = memoryview(byte_array[1:]).cast(array_dtype)
        return cls(*view)

            
    def __init__(self, x, y) -> None:
        self.__x = float(x)
        self.__y = float(y)
    
    # using properties we make our attributes readonly, as we dont provide a setter!
    # its just a getter only property!
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    def __bytes__(self):
        # since we need bytes, we specify the bytetype and simply add it to the data section, after all bytes are like strings!
        # also since we implemented __iter__, our self is an iterable and we can easily pass it to bytes!
        # but, note that we dont want single byte values, but rather an array representation. so its ok to use array
        # however we could also use a list or tuple, or ourselves as we are iterable, but if we do use bytes(self)
        # we will endup in an infinit recursion! (cuz to get the bytes, we would recursively do bytes(self) and it would
        # call __bytes__ with has bytes(self) and this goes on and on )
        # if we do bytes(tuple(self)) the previous error goes away, but bytes only accepts ints and not floats! 
        # so we need to do a conversion!
        # however, this also converts the lements only, without any structure. we can use array and 
        # store our data with the array structure. 
        # so in short 
        # The key difference here is that the first expression creates a byte representation of an array, 
        # including the type information specified by self.typecode, 
        # while the second expression creates a byte representation of the individual elements in self, without any specific data structure.
        # Depending on our specific use case and requirements, we can choose the appropriate expression. 
        # If we need to preserve the array structure and type information, 
        # the first expression with array would be more suitable as for loading it later on
        # we can simply use array() to create new array and acccess the values readily, 
        # 
        # If we only need to convert the elements to bytes individually, 
        # the second expression with the generator expression would be sufficient but 
        # for loading it back, we have to write more code to extract our values and account 
        # for more stuff
        # also note that bytes accepts int as well and not only iterables, if we feed it an int, it will
        # create an empty bytearray as long as that given int, so down below, when we are converting typecode
        # to bytes, make sure we wrap it in [], so we convert the actual typecode and not create an empty bytes array!!
        # 
        print(f'typecode: {self.typecode}')
        arr =array(self.typecode, self)
        # print(f'array of bytes: {arr} {arr.tobytes()}')
        # print(f'converted using bytes: {bytes(array(self.typecode, self))}')
        # print(f'bytes of bytecode: {bytes([ord(self.typecode)])} end')
        return (bytes([ord(self.typecode)]) + bytes(array(self.typecode, self)))
    
    def __str__(self) -> str:
        return str(tuple(self))
    
    def __repr__(self) -> str:
        # we specifically use self, so if in the future we inherit from this class,
        # the children actually instantiate their own class and not their parent!
        cls = self.__class__.__name__
        return f"{cls}({repr(self.x)}, {repr(self.y)})"
    
    def __hash__(self) -> int:
        # instead of hashing the two values, we get the hash of their tuple as tuple is 
        # imuutable, its hash is also unique, 
        # if we wanted to not use the tuple, we had to do a xor on all values to get a
        # single hash like hash(self.x) ^ hash(self.y)
        return hash(tuple(self.x, self.y))
    
    # without this, bool(instance) will always return true, as it goes for the refrence! but 
    # here we are making sure we want the status of the vector emptiness. if its not empty 
    # send true, otherwise false. since we calculate vectors magnitude, we can use its __abs__
    # here, anything otherthan 0 means its filled with values and thus returns True
    def __bool__(self):
        return bool(abs(self))
    
    # we use this to return the magnitude of our vector
    def __abs__(self):
        return math.hypot(self.x, self.y)
    
    # we were also instructed to make this iterable, since we have two points
    # we need to return them one by one, implementing it like this frees us from 
    # implementing __next__ for __iter__, as its a generator here and manages everything itself
    # by implementing __iter__, now our class also supports in, for, iter, etc
    def __iter__(self):
        return (x for x in (self.x, self.y))
    
    # in order to have proper hash, we need to have __eq__ implemented as well
    # since we already implemented __abs__, we have the vector magnitude, so we can 
    # use that instead of individually checking the attributes of the two object
    # and also __eq__ is ==, which checks equality of values in the object
    # we could also do a tuple(self) == tuple(other) apparently!
    # which seems like the correct/better answer?, cuz in our answer a vector of 
    #v(2,3) will be the same of v(3,2) cuz their magnitude is the same, but they are not the same?
    # and tuple(2,3) captures this meaning as well? however
    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other)
    
    # lets add support for indexes and slices (basically, sequence)
    # def __getitem__(self, idx):
    #     return 
    
    
if __name__ == "__main__":
    v = Vector2d(3,4)
    v2 = Vector2d(0,0)
    v3=Vector2d(4,3)
    v_clone = eval(repr(v))
    print(f'v = {v}')
    print(f'v_clone = {v_clone}')
    print(f'v2 = {v2}')
    print(f'v3 = {v3}')
    a,b = v 
    print(f'v.x={a} v.y={b}')
        
    it = iter(v)
    print(f'next(it): {next(it)}')
    print(f'next(it): {next(it)}')
    print(f'bool(v_clone): {bool(v_clone)}, bool(v2): {bool(v2)}')
    print(f'3 is in v? :{3 in v}')
    print(f'30 is in v? :{30 in v}')
    bytes_rep = bytes(v)
    print(bytes_rep, len(bytes_rep))
    print(f'v == v_clone: {v == v_clone}')
    print(f'v == v2: {v == v2}')
    print(f'v == v3: {v == v3}')
    print(f'abs(v): {abs(v)} abs(v3): {abs(v3)}')
    print(f'bytes(v3): {Vector2d.from_bytes(bytes(v3))}')
    