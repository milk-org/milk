#if !defined(GENIMAGE_H)
#define GENIMAGE_H

void __attribute__((constructor)) libinit_image_gen();

/** @brief creates a double star */
imageID make_double_star(const char *ID_name,
                         uint32_t    l1,
                         uint32_t    l2,
                         double      intensity_1,
                         double      intensity_2,
                         double      separation,
                         double      position_angle);

/** @brief  creates a disk */
imageID make_disk(const char *ID_name,
                  uint32_t    l1,
                  uint32_t    l2,
                  double      x_center,
                  double      y_center,
                  double      radius);

/** @brief  creates a disk */
imageID make_subpixdisk(const char *ID_name,
                        uint32_t    l1,
                        uint32_t    l2,
                        double      x_center,
                        double      y_center,
                        double      radius);

/** @brief creates a shape with contour described by sum of sine waves */
imageID make_subpixdisk_perturb(const char *ID_name,
                                uint32_t    l1,
                                uint32_t    l2,
                                double      x_center,
                                double      y_center,
                                double      radius,
                                long        n,
                                double     *ra,
                                double     *ka,
                                double     *pa);

/** @brief  creates a square */
imageID make_square(const char *ID_name,
                    uint32_t    l1,
                    uint32_t    l2,
                    double      x_center,
                    double      y_center,
                    double      radius);

imageID make_rectangle(const char *ID_name,
                       uint32_t    l1,
                       uint32_t    l2,
                       double      x_center,
                       double      y_center,
                       double      radius1,
                       double      radius2);

imageID make_line(const char *IDname,
                  uint32_t    l1,
                  uint32_t    l2,
                  double      x1,
                  double      y1,
                  double      x2,
                  double      y2,
                  double      t);

/** @brief draw line crossing point xc, yc with angle, pixel value is coordinate axis perp to line */
imageID make_lincoordinate(const char *IDname,
                           uint32_t    l1,
                           uint32_t    l2,
                           double      x_center,
                           double      y_center,
                           double      angle);

imageID make_hexagon(const char *IDname,
                     uint32_t    l1,
                     uint32_t    l2,
                     double      x_center,
                     double      y_center,
                     double      radius);

imageID IMAGE_gen_segments2WFmodes(const char *prefix,
                                   long        ndigit,
                                   const char *IDout_name);

imageID make_hexsegpupil(
    const char *IDname, uint32_t size, double radius, double gap, double step);

imageID make_jacquinot_pupil(const char *ID_name,
                             uint32_t    l1,
                             uint32_t    l2,
                             double      x_center,
                             double      y_center,
                             double      width,
                             double      height);

imageID make_sectors(const char *ID_name,
                     uint32_t    l1,
                     uint32_t    l2,
                     double      x_center,
                     double      y_center,
                     double      step,
                     long        NB_sectors);

imageID
make_rnd(const char *ID_name, uint32_t l1, uint32_t l2, const char *options);

imageID make_rnd_double(const char *ID_name,
                        uint32_t    l1,
                        uint32_t    l2,
                        const char *options);
/*int make_rnd1(const char *ID_name, long l1, long l2, const char *options);*/

imageID
make_gauss(const char *ID_name, uint32_t l1, uint32_t l2, double a, double A);

imageID make_FiberCouplingOverlap(const char *ID_name);

imageID make_2axis_gauss(const char *ID_name,
                         uint32_t    l1,
                         uint32_t    l2,
                         double      a,
                         double      A,
                         double      E,
                         double      PA);

imageID make_cluster(const char *ID_name,
                     uint32_t    l1,
                     uint32_t    l2,
                     const char *options);

imageID make_galaxy(const char *ID_name,
                    uint32_t    l1,
                    uint32_t    l2,
                    double      S_radius,
                    double      S_L0,
                    double      S_ell,
                    double      S_PA,
                    double      E_radius,
                    double      E_L0,
                    double      E_ell,
                    double      E_PA);

imageID make_Egalaxy(const char *ID_name,
                     uint32_t    l1,
                     uint32_t    l2,
                     const char *options);

/** @brief  make image of EZ disk */
imageID gen_image_EZdisk(const char *ID_name,
                         uint32_t    size,
                         double      InnerEdge,
                         double      Index,
                         double      Incl);

imageID make_slopexy(
    const char *ID_name, uint32_t l1, uint32_t l2, double sx, double sy);

imageID
make_dist(const char *ID_name, uint32_t l1, uint32_t l2, double f1, double f2);

imageID make_PosAngle(
    const char *ID_name, uint32_t l1, uint32_t l2, double f1, double f2);

imageID make_psf_from_profile(const char *profile_name,
                              const char *ID_name,
                              uint32_t    l1,
                              uint32_t    l2);

imageID make_offsetHyperGaussian(
    uint32_t size, double a, double b, long n, const char *IDname);

imageID
make_cosapoedgePupil(uint32_t size, double a, double b, const char *IDname);

imageID make_2Dgridpix(const char *IDname,
                       uint32_t    xsize,
                       uint32_t    ysize,
                       double      pitchx,
                       double      pitchy,
                       double      offsetx,
                       double      offsety);

imageID make_tile(const char *IDin_name, uint32_t size, const char *IDout_name);

imageID
image_gen_im2coord(const char *IDin_name, uint8_t axis, const char *IDout_name);

imageID
image_gen_make_voronoi_map(const char *filename,
                           const char *IDout_name,
                           uint32_t    xsize,
                           uint32_t    ysize,
                           float radius, // maximum radius of each Voronoi zone
                           float maxsep  // gap between Voronoi zones
                          );

#endif
