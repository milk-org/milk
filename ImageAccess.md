# Accessing Images and streams

## Type ImageStreamIO

All images and streams are stored using in the ImageStreamIO format. The format supports images in shared memory (streams) as well as images in local memory.

## Type IMGID

milk uses the IMGID struture to hold images.
This structure is one level above ImageStreamIO, and is local to the milk process (not included in shared memory).

IMGID is the preferred way to call images and streams as function arguments, usually passed by pointer.

IMGID core functions are provided as static inline type in IMGID.h.


## Image array and imageID

milk-CLI holds an array of images in memory, so that they can be called by name in CLI mode. The array of images is data.image, with number of elements data.NB_MAX_IMAGE. Images that are in local memory can then quickly be accessed by their index ID, an integer (type imageID). In this mode, IMGID holds the index value IMGID.ID.

imageID core functions are provided as static inline type in imageID.h.


## Coding practices

Images and streams should be accessed as IMGID as much as possible. ImageID and image array features should only be used in milk-CLI mode to expose images and streams to the CLI, and should not be used in fps-standalone implementation.

A CLI wrapper to a function will use names (strings) as arguments.



## IMGID core functions

Functions are in IMGID.h.

### Creating an IMGID

IMPORTANT: Creating an IMGID does not create or allocate the image data.

Creating a blank IMGID:
```c
static inline IMGID imgid_make()
```


Creating an IMGID with name. Special character ">" can be used to set some of the fields. For example, "s>tf64>im1" sets image name to "im1", shared memory, and type float64.
```c
static inline IMGID imgid_make_from_name(CONST_WORD  name)
```

Creating a named 2D IMGID:
```c
static inline IMGID imgid_make_from_name_2D(
    CONST_WORD name,
    uint32_t xsize,
    uint32_t ysize
)
```

Creating a named 3D IMGID:
```c
static inline IMGID imgid_make_from_name_3D(
    CONST_WORD name,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
```

### Copy an IMGID

Copy fields from one IMGID to another one:
```c
static inline int imgid_copy(
    IMGID *imgin,
    IMGID *imgout
)
```
Note that the name is not copied.


### Comparing IMGIDs

Comparing fields:
```c
static inline uint64_t imgid_compare(
    IMGID img,
    IMGID imgtemplate
)
```

Comparing templates:
```c
static inline uint64_t imgid_compare_md(
    IMGID img,
    IMGID imgtemplate
)
```

### Updating IMGID

```c
static inline errno_t imgid_update_creationparams(IMGID *img)
```



### Loading images from shared memory to IMGID

To load an image from shared memory, use the function:

```c
static inline IMGID imgid_connect(CONST_WORD sname, IMGID *img, int FLAG)
```

If successful, IMGID.ID is set to 0 and the IMGID fields point to the image and its metadata. If not successful, IMGID.ID is set to -1.











## imageID functions

These functions are to manage image access with an array of images, such as in milk-CLI mode.

Functions are static inline type in COREMOD_memory/imageID.h

### Registering an IMGID in the image array

```c
imageID registerIMGID(
    IMGID *img,
    IMAGE *imagearray,
    long NB_images
)
```

The function registered the IMGID as an entry in the imagearray, and returns the imageID index to which the IMGID has been registered in the array.



### Resolving an imageID from image name

```c
static inline imageID resolveIMGID(
    IMGID *img,
    int ERRMODE,
    IMAGE *imagearray,
    long NB_images
)
```

When an image is accessed by its name, the function resolveIMGID() will look for image's ID corresponding the IMGID's name field (this is called "resolving" the image), and writes it to IMGID.ID. If the image has already been resolved, then the function quickly returns its ID.


## Use cases, examples

### Reading a stream

To read an ImageStreamIO stream:

```c
// Create IMGID
IMGID img1 = imgid_make();
// Connect to stream
imgid_connect("streamname1", &img1, 0);
if(img1.ID == -1) {
	// handle failure to load stream
}
```

### Creating an image in local memory

The preferred way is to use function imgid_make_from_name. Note that the image name may not be used if in non-CLI mode, and multiple images may have the same name.
In CLI mode, the image will be accessed by name, so it is imperative to avoid duplicates.

```c
// Create IMGID
IMGID img = imgid_make_from_name("im1")
img.naxis = 2;
img.size[0] = 128;
img.size[1] = 128;

// Create image and allocate image memory
imgid_mkimage(&img);
```

### Creating a stream

Creating a stream may overwrite an existing stream. If that is OK and no check is necessary, use:

```c
// Create IMGID
IMGID img = imgid_make_from_name("im1")
img.naxis = 2;
img.size[0] = 128;
img.size[1] = 128;
img.shared = 1;

// Create image and allocate image memory
imgid_mkimage(&img);
```

### Connecting to a stream with checks

Reading a stream, checking if it has the correct format (size, type).

#### Fail mode

If format check passes, connection is successful: use the existing stream.
If format check fails, do not connect, and set img1.ID to -1.

```c
// Create IMGID
IMGID img1 = imgid_make();
img.naxis = 2;
img.size[0] = 128;
img.size[1] = 128;

// Connect to stream
// Test if datatype, naxis, size, type, NBkw and CBsize match
// fail if not matching
imgid_connect("streamname1", &img1, IMGID_CONNECT_CHECK_FAIL);
```

#### Create mode

If format check passes, connection is successful: use the existing stream.
If format check fails, re-create the stream to the correct format.

```c
// Connect to stream
// Test if datatype, naxis, size, type, NBkw and CBsize match
// Create new stream if not matching
imgid_connect("streamname1", &img1, IMGID_CONNECT_CHECK_CREATE);
```
