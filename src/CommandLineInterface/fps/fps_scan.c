/**
 * @file    fps_scan.c
 * @brief   scan and load FPSs
 */

#include <dirent.h>
#include <libgen.h>   // basename
#include <sys/stat.h> // fstat

#include <regex.h>

#include "CommandLineInterface/CLIcore.h"

#include "TUItools.h"

#include "fps_connect.h"
#include "fps_disconnect.h"

/** @brief scan and load FPSs
 *
 */

errno_t functionparameter_scan_fps(
    uint32_t                   mode,
    char                      *fpsnamemask,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE         *keywnode,
    int                       *ptr_NBkwn,
    int                       *ptr_fpsindex,
    long                      *ptr_pindex,
    int                        verbose
)
{
    int fpsindex;
    int pindex;
    int kwnindex;
    int NBkwn;
    int l;


    // FPS list file
    FILE *fpfpslist = NULL;
    int   fpslistcnt = 0;
    char  FPSlist[NB_FPS_MAX][200] = {0};

    // Static variables
    static int  shmdirname_init = 0;
    static char shmdname[STRINGMAXLEN_SHMDIRNAME] = {0};



    for(int kindex = 0; kindex < NB_KEYWNODE_MAX; kindex++)
    {
        keywnode[kindex].NBchild = 0;
    }


    // scan filesystem for fps entries

    if(verbose > 0)
    {
        printf(
            "\n\n\n====================== SCANNING FPS ON SYSTEM "
            "==============================\n\n");
        fflush(stdout);
    }

    if(shmdirname_init == 0)
    {
        function_parameter_struct_shmdirname(shmdname);
        shmdirname_init = 1;
    }


    // disconnect previous fps
    for(fpsindex = 0; fpsindex < NB_FPS_MAX; fpsindex++)
    {
        if(fps[fpsindex].SMfd > -1)  // connected
        {
            function_parameter_struct_disconnect(&fps[fpsindex]);
        }
    }


    // request match to file ./fpscomd/fpslist.txt
    if(mode & 0x0001)
    {
        if((fpfpslist = fopen("fpscmd/fpslist.txt", "r")) != NULL)
        {
            char   *FPSlistline = NULL;
            size_t  len         = 0;
            ssize_t read;

            while((read = getline(&FPSlistline, &len, fpfpslist)) != -1)
            {
                if(FPSlistline[0] != '#')
                {
                    char *pch;

                    pch = strtok(FPSlistline, " \t\n\r");
                    if(pch != NULL)
                    {
                        snprintf(FPSlist[fpslistcnt], 200,
                                 "%s", pch);
                        fpslistcnt++;
                    }
                }
            }
            fclose(fpfpslist);
        }
        else
        {
            if(verbose > 0)
            {
                printf("Cannot open file fpscmd/fpslist.txt\n");
            }
        }

        int fpsi;
        for(fpsi = 0; fpsi < fpslistcnt; fpsi++)
        {
            if(verbose > 0)
            {
                printf("FPSname must match %s\n", FPSlist[fpsi]);
            }
        }
    }

    //    NBparam = function_parameter_struct_connect(fpsname, &fps[fpsindex]);

    // create ROOT node (invisible)
    keywnode[0].keywordlevel = 0;
    snprintf(keywnode[0].keyword[0], 64, "ROOT");
    keywnode[0].leaf    = 0;
    keywnode[0].NBchild = 0;
    NBkwn               = 1;


    // scan directory for FPS files
    //
    DIR           *d;
    struct dirent *dir;
    d = opendir(shmdname);
    if(d)
    {
        fpsindex = 0;
        pindex   = 0;
        while(((dir = readdir(d)) != NULL))
        {
            char *pch = strstr(dir->d_name, ".fps.shm");

            int matchOK = 0;

            // name filtering
            // if mask is "_ALL", display all
            //
            if(strcmp(fpsnamemask, "_ALL") == 0)
            {
                matchOK = 1;
            }
            else
            {
                if(strncmp(dir->d_name, fpsnamemask, strlen(fpsnamemask)) == 0)
                {
                    matchOK = 1;
                }
            }


            if(mode & 0x0001)  // enforce match to list
            {
                int matchOKlist = 0;
                int fpsi;

                for(fpsi = 0; fpsi < fpslistcnt; fpsi++)
                {
                    regex_t reegex;

                    regcomp(&reegex, FPSlist[fpsi], REG_EXTENDED | REG_NEWLINE);
                    int rmatch = regexec(&reegex, dir->d_name, 0, NULL, 0);

                    if(rmatch == 0)
                    {
                        matchOKlist = 1;
                    }
                }

                matchOK *= matchOKlist;
            }


            if((pch) && (matchOK == 1))
            {

                // is file sym link ?
                struct stat buf;
                int         retv;
                char        fullname[STRINGMAXLEN_FULLFILENAME];
                char        shmdname[STRINGMAXLEN_DIRNAME];
                function_parameter_struct_shmdirname(shmdname);

                WRITE_FULLFILENAME(fullname, "%s/%s", shmdname, dir->d_name);

                retv = lstat(fullname, &buf);
                if(retv == -1)
                {
                    TUI_exit();
                    printf("File \"%s\"", dir->d_name);
                    perror("Error running lstat on file ");
                    printf("File %s line %d\n", __FILE__, __LINE__);
                    fflush(stdout);
                    exit(EXIT_FAILURE);
                }

                if(S_ISLNK(buf.st_mode))  // resolve link name
                {
                    char fullname[STRINGMAXLEN_FULLFILENAME];
                    char linknamefull[STRINGMAXLEN_FULLFILENAME];
                    char linkname[STRINGMAXLEN_FULLFILENAME];

                    char shmdname[STRINGMAXLEN_DIRNAME];
                    function_parameter_struct_shmdirname(shmdname);

                    //fps_symlink[fpsindex] = 1;
                    if(snprintf(fullname,
                                STRINGMAXLEN_FULLFILENAME,
                                "%s/%s",
                                shmdname,
                                dir->d_name) < 0)
                    {
                        PRINT_ERROR("snprintf error");
                    }

                    if(readlink(fullname, linknamefull, 200 - 1) == -1)
                    {
                        // todo: replace with realpath()
                        PRINT_ERROR("readlink() error");
                    }
                    strcpy(linkname, basename(linknamefull));

                    int          lOK = 1;
                    unsigned int ii  = 0;
                    while((lOK == 1) && (ii < strlen(linkname)))
                    {
                        if(linkname[ii] == '.')
                        {
                            linkname[ii] = '\0';
                            lOK          = 0;
                        }
                        ii++;
                    }

                    //                        strncpy(streaminfo[sindex].linkname, linkname, nameNBchar);
                }
                //else {
                //  fps_symlink[fpsindex] = 0;
                //}

                //fps_symlink[fpsindex] = 0;

                char fpsname[STRINGMAXLEN_FPS_NAME];
                long strcplen  = strlen(dir->d_name) - strlen(".fps.shm");
                int  strcplen1 = STRINGMAXLEN_FPS_NAME - 1;
                if(strcplen < strcplen1)
                {
                    strcplen1 = strcplen;
                }

                strncpy(fpsname, dir->d_name, strcplen1);
                fpsname[strcplen1] = '\0';

                if(verbose > 0)
                {
                    printf(
                        "FOUND FPS %s - (RE)-CONNECTING  "
                        "[%d]\n",
                        fpsname,
                        fpsindex);
                    fflush(stdout);
                }


                long NBparamMAX =
                    function_parameter_struct_connect(fpsname,
                                                      &fps[fpsindex],
                                                      FPSCONNECT_SIMPLE);



                // FILTERING
                int fpskeep = 1; // 0 if not kept

                // name
                char *fps_filtstring_name = getenv("FPS_FILTSTRING_NAME");
                if(fps_filtstring_name)
                {
                    char *ptr = strstr(fps[fpsindex].md->name, fps_filtstring_name);
                    if(ptr == NULL)  // string not found
                    {
                        fpskeep = 0;
                    }
                }


                // keyword
                char *fps_filtstring_keyword = getenv("FPS_FILTSTRING_KEYWORD");
                if(fps_filtstring_keyword)
                {
                    char *ptr = strstr(fps[fpsindex].md->keywordarray, fps_filtstring_keyword);
                    if(ptr == NULL)  // string not found
                    {
                        fpskeep = 0;
                    }
                }



                // call function
                char *fps_filtstring_callfunc = getenv("FPS_FILTSTRING_CALLFUNC");
                if(fps_filtstring_callfunc)
                {
                    char *ptr = strstr(fps[fpsindex].md->callfuncname, fps_filtstring_callfunc);
                    if(ptr == NULL)  // string not found
                    {
                        fpskeep = 0;
                    }
                }


                // module name
                char *fps_filtstring_module = getenv("FPS_FILTSTRING_MODULE");
                if(fps_filtstring_module)
                {
                    int modulematch = 0;
                    for(int mdindex = 0; mdindex < FPS_MAXNB_MODULE; mdindex++)
                    {
                        if(strlen(fps[fpsindex].md->modulename[mdindex]) > 1)
                        {
                            char *ptr = strstr(fps[fpsindex].md->modulename[mdindex],
                                               fps_filtstring_module);
                            if(ptr != NULL)  // string found
                            {
                                modulematch = 1;
                            }
                        }

                    }
                    if(modulematch == 0)
                    {
                        fpskeep = 0;
                    }
                }





                if(fpskeep == 1)
                {


                    long pindex0;
                    for(pindex0 = 0; pindex0 < NBparamMAX; pindex0++)
                    {
                        if(fps[fpsindex].parray[pindex0].fpflag &
                                FPFLAG_ACTIVE) // if entry is active
                        {
                            // find or allocate keyword node
                            int level;
                            for(level = 1;
                                    level <
                                    fps[fpsindex].parray[pindex0].keywordlevel + 1;
                                    level++)
                            {

                                // does node already exist ?
                                int scanOK = 0;
                                // scan existing nodes looking for match
                                for(kwnindex = 0; kwnindex < NBkwn; kwnindex++)
                                {
                                    if(keywnode[kwnindex].keywordlevel ==
                                            level) // levels have to match
                                    {
                                        int match = 1;
                                        // keywords at all levels need to match

                                        for(l = 0; l < level; l++)
                                        {
                                            if(strcmp(fps[fpsindex]
                                                      .parray[pindex0]
                                                      .keyword[l],
                                                      keywnode[kwnindex]
                                                      .keyword[l]) != 0)
                                            {
                                                match = 0;
                                            }
                                            //  printf("TEST MATCH : %16s %16s  %d\n", fps[fpsindex].parray[i].keyword[l], keywnode[kwnindex].keyword[l], match);
                                        }
                                        if(match == 1)  // we have a match
                                        {
                                            scanOK = 1;
                                        }
                                        //             printf("   -> %d\n", scanOK);
                                    }
                                }

                                if(scanOK == 0)  // node does not exit -> create it
                                {
                                    // look for parent
                                    int scanparentOK = 0;
                                    int kwnindexp    = 0;
                                    keywnode[kwnindex].parent_index =
                                        0; // default value, not found -> assigned to ROOT

                                    while((kwnindexp < NBkwn) &&
                                            (scanparentOK == 0))
                                    {
                                        if(keywnode[kwnindexp].keywordlevel ==
                                                level - 1) // check parent has level-1
                                        {
                                            int match = 1;

                                            for(
                                                l = 0; l < level - 1;
                                                l++) // keywords at all levels need to match
                                            {
                                                if(strcmp(fps[fpsindex]
                                                          .parray[pindex0]
                                                          .keyword[l],
                                                          keywnode[kwnindexp]
                                                          .keyword[l]) != 0)
                                                {
                                                    match = 0;
                                                }
                                            }
                                            if(match == 1)  // we have a match
                                            {
                                                scanparentOK = 1;
                                            }
                                        }
                                        kwnindexp++;
                                    }

                                    if(scanparentOK == 1)
                                    {
                                        keywnode[kwnindex].parent_index =
                                            kwnindexp - 1;
                                        int cindex;
                                        cindex = keywnode[keywnode[kwnindex]
                                                          .parent_index]
                                                 .NBchild;
                                        keywnode[keywnode[kwnindex].parent_index]
                                        .child[cindex] = kwnindex;
                                        keywnode[keywnode[kwnindex].parent_index]
                                        .NBchild++;
                                    }

                                    if(verbose > 0)
                                    {
                                        printf(
                                            "CREATI"
                                            "NG "
                                            "NODE "
                                            "%d ",
                                            kwnindex);
                                    }
                                    keywnode[kwnindex].keywordlevel = level;

                                    for(l = 0; l < level; l++)
                                    {
                                        char tmpstring[200];
                                        strcpy(keywnode[kwnindex].keyword[l],
                                               fps[fpsindex]
                                               .parray[pindex0]
                                               .keyword[l]);
                                        //printf(" %s", keywnode[kwnindex].keyword[l]);
                                        if(l == 0)
                                        {
                                            strcpy(keywnode[kwnindex].keywordfull,
                                                   keywnode[kwnindex].keyword[l]);
                                        }
                                        else
                                        {
                                            snprintf(tmpstring,
                                                     200,
                                                     ".%s",
                                                     keywnode[kwnindex].keyword[l]);
                                            strcat(keywnode[kwnindex].keywordfull,
                                                   tmpstring);
                                        }
                                    }
                                    if(verbose > 0)
                                    {
                                        printf(
                                            "   %d "
                                            "%d\n",
                                            keywnode[kwnindex].keywordlevel,
                                            fps[fpsindex]
                                            .parray[pindex0]
                                            .keywordlevel);
                                    }

                                    if(keywnode[kwnindex].keywordlevel ==
                                            fps[fpsindex].parray[pindex0].keywordlevel)
                                    {
                                        //									strcpy(keywnode[kwnindex].keywordfull, fps[fpsindex].parray[i].keywordfull);

                                        keywnode[kwnindex].leaf     = 1;
                                        keywnode[kwnindex].fpsindex = fpsindex;
                                        keywnode[kwnindex].pindex   = pindex0;
                                    }
                                    else
                                    {

                                        keywnode[kwnindex].leaf     = 0;
                                        keywnode[kwnindex].fpsindex = fpsindex;
                                        keywnode[kwnindex].pindex   = 0;
                                    }

                                    kwnindex++;
                                    NBkwn = kwnindex;
                                }
                            }
                            pindex++;
                        }
                    }
                }

                if(verbose > 0)
                {
                    printf("--- FPS %4d  %-20s %ld parameters\n",
                           fpsindex,
                           fpsname,
                           fps[fpsindex].md->NBparamMAX);
                }

                fpsindex++;
            }
        }
        closedir(d);
    }
    else
    {
        char shmdname[STRINGMAXLEN_SHMDIRNAME];
        function_parameter_struct_shmdirname(shmdname);
        printf("ERROR: missing %s directory\n", shmdname);
        printf("File %s line %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(EXIT_FAILURE);
    }

    if(verbose > 0)
    {
        printf(
            "\n\n=================[END] SCANNING FPS ON SYSTEM [END]=  %d  "
            "========================\n\n\n",
            fpsindex);
        fflush(stdout);
    }

    *ptr_NBkwn    = NBkwn;
    *ptr_fpsindex = fpsindex;
    *ptr_pindex   = pindex;

    return RETURN_SUCCESS;
}
