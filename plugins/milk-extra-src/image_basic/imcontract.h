/** @file imcontract.h
 */

errno_t imcontract_addCLIcmd();

imageID
basic_contract(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_contract3D(
    const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);
