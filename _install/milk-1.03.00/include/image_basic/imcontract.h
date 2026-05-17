/**
 * @file imcontract.h
 * @brief Imcontract module
 */

/** @file imcontract.h
 */

errno_t CLIADDCMD_image_basic__imcontract();

imageID
basic_contract(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_contract3D(
    const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);
