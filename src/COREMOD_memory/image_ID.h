/**
 * @file    image_ID.h
 */

imageID image_ID(const char *name);

imageID image_ID_noaccessupdate(const char *name);

imageID next_avail_image_ID(imageID preferredID);
