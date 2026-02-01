/**
 * @file    image_ID.h
 */

imageID image_ID(const char *name, IMAGE *imagearray, long NB_images);

imageID image_ID_noaccessupdate(const char *name, IMAGE *imagearray, long NB_images);

imageID next_avail_image_ID(imageID preferredID);
