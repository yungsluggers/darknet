#include "darknet.h"

#include <sys/time.h>
#include <assert.h>
#include <string.h>

float *get_regression_values(char **labels, int n)
{
    float *v = calloc(n, sizeof(float));
    int i;
    for (i = 0; i < n; ++i)
    {
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p + 1);
    }
    return v;
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *label, int top, char *filepath, char *size)
{
    int sizeInt = atoi(size);
    //printf("value of imagedata: %.*s\n", (int)sizeof(imagedata) + 7, imagedata);
    // converting string ex: 123,242,234,234 to int array
    int totalsize = sizeInt * sizeInt * 4;
    int imgIntArray[90000] = {};

    char imagedata[999999];
    FILE *f = fopen(filepath, "r");
    fscanf(f, "%[^\n]", imagedata);
    fclose(f);

    //memset(imgIntArray, 0, totalsize * sizeof(int));
    char *tok = strtok(imagedata, ",");
    int i = 0;
    // Keep going until we run out of tokens
    while (tok)
    {
        //printf(tok);
        //printf("\n");
        // Don't overflow your target array
        if (i < totalsize)
        {
            // Convert to integer and store it
            imgIntArray[i++] = atoi(tok);
        }
        // Get the next token from the string - note the use of NULL
        // instead of the string in this case - that tells it to carry
        // on from where it left off.
        tok = strtok(NULL, ",");
    }

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    //list *options = read_data_cfg(datacfg);

    char *name_list = "data/9k.labels";
    //if(top == 0) top = option_find_int(options, "top", 1);

    //printf( "value of namelist: %s\n", name_list );

    char **n = get_labels(name_list);

    i = 0;
    char *blah = label;
    while (*n != NULL)
    {
        //printf("value of n: %.*s\n", (int)sizeof(*n), *n);
        if (!strcmp(*n, blah))
        {
            //printf("found n: %.*s\n", (int)sizeof(*n) + 1, *n);
            break;
        }
        i++;
        *n++;
    }

    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    image im = load_image_bitmap(imgIntArray, sizeInt, sizeInt);

    // for (int i=0; i < 99; i++) {
    //     printf("%lf\n",im.data[i]);
    // }
    image r = letterbox_image(im, net->w, net->h);
    //image r = resize_min(im, 320);
    //printf("%d %d\n", r.w, r.h);
    resize_network(net, r.w, r.h);
    //printf("%d %d\n", r.w, r.h);
    float *X = r.data;
    time = clock();
    float *predictions = network_predict(net, X);
    if (net->hierarchy)
        hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
    top_k(predictions, net->outputs, top, indexes);
    //fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
    char **labels = get_labels("data/9k.labels");
    char **names = get_labels("data/9k.names");
    for (int j = 0; j < 5; j++)
    {
        int index = indexes[j];
        //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
        //else printf("%s: %f\n",names[index], predictions[index]);
        printf("%.8f: %s: %s\n", predictions[index], labels[index], names[index]);
        printf("selected: %s\n", names[i]);
    }
    if (r.data != im.data)
        free_image(r);
    free_image(im);
    //if (filename) break;*/
}

void one_label_classifier(char *datacfg, char *cfgfile, char *weightfile, char *label, int top, char *filepath, char *size)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char filepathbuff[1024];
    char *filepathinput = filepathbuff;

    char idbuff[256];
    char *idinput = idbuff;

    int imgIntArray[90000] = {};

    char imagedata[999999];

    while (1)
    {
        if (filepath)
        {
            strncpy(filepathinput, filepath, 1024);
            strncpy(idinput, label, 256);
        }
        else
        {
            //printf("Enter Image Path and id: \n");
            fflush(stdout);
            scanf("%1024s %256s", filepathinput, idinput);
            if (!filepathinput)
                return;
            strtok(filepathinput, "\n");
            if (!idinput)
                return;
            strtok(idinput, "\n");
        }

        int sizeInt = atoi(size);
        //printf("value of imagedata: %.*s\n", (int)sizeof(imagedata) + 7, imagedata);
        // converting string ex: 123,242,234,234 to int array
        int totalsize = sizeInt * sizeInt * 4;

        memset(imagedata, 0, sizeof imagedata);

        FILE *f = fopen(filepathinput, "r");
        fscanf(f, "%[^\n]", imagedata);
        fclose(f);

        char *tok = strtok(imagedata, ",");
        int i = 0;
        // Keep going until we run out of tokens
        while (tok)
        {
            // Don't overflow your target array
            if (i < totalsize)
            {
                // Convert to integer and store it
                imgIntArray[i++] = atoi(tok);
                printf("%d\n", atoi(tok));
            }
            // Get the next token from the string - note the use of NULL
            // instead of the string in this case - that tells it to carry
            // on from where it left off.
            tok = strtok(NULL, ",");
        }

        //printf( "value of namelist: %s\n", name_list );

        char **n = get_labels("data/9k.labels");

        i = 0;
        char *blah = idinput;
        while (*n != NULL)
        {
            //printf("value of n: %.*s\n", (int)sizeof(*n), *n);
            if (!strcmp(*n, blah))
            {
                //printf("found n: %.*s\n", (int)sizeof(*n) + 1, *n);
                break;
            }
            i++;
            *n++;
        }

        int *indexes = calloc(top, sizeof(int));

        image im = load_image_bitmap(imgIntArray, sizeInt, sizeInt);

        // for (int i=0; i < 99; i++) {
        //     printf("%lf\n",im.data[i]);
        // }

        image r = letterbox_image(im, net->w, net->h);
        //image r = resize_min(im, 320);
        //printf("%d %d\n", r.w, r.h);
        resize_network(net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);
        float *X = r.data;
        float *predictions = network_predict(net, X);
        if (net->hierarchy)
            hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        //fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        printf("%.8f\n", predictions[i]);
        if (r.data != im.data)
            free_image(r);
        free_image(im);
        if (filepath)
            break;
    }
}

void run_classifier(int argc, char **argv)
{
    if (argc < 4)
    {
        //fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 10);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *size = (argc > 6) ? argv[6] : 0;
    char *label = (argc > 7) ? argv[7] : 0;
    char *filepath = (argc > 8) ? argv[8] : 0;
    if (0 == strcmp(argv[2], "predict"))
        predict_classifier(data, cfg, weights, label, top, filepath, size);
    if (0 == strcmp(argv[2], "one_label"))
        one_label_classifier(data, cfg, weights, label, top, filepath, size);
    // else if(0==strcmp(argv[2], "fout")) file_output_classifier(data, cfg, weights, filename);
    // else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
    // else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear);
    // else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
    // else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
    // else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    // else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, layer);
    // else if(0==strcmp(argv[2], "csv")) csv_classifier(data, cfg, weights);
    // else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
    // else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights);
    // else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    // else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    // else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    // else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);
}
