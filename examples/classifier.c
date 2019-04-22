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

void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network *net = load_network(filename, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m / 1000;
    int num = (i + 1) * m / splits - i * m / splits;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for (i = 1; i <= splits; ++i)
    {
        time = clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i + 1) * m / splits - i * m / splits;
        char **part = paths + (i * m / splits);
        if (i != splits)
        {
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock() - time));

        time = clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc / i, topk, avg_topk / i, sec(clock() - time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for (i = 0; i < m; ++i)
    {
        int class = -1;
        char *path = paths[i];
        for (j = 0; j < classes; ++j)
        {
            if (strstr(path, labels[j]))
            {
                class = j;
                break;
            }
        }
        int w = net->w;
        int h = net->h;
        int shift = 32;
        image im = load_image_color(paths[i], w + shift, h + shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = calloc(classes, sizeof(float));
        for (j = 0; j < 10; ++j)
        {
            float *p = network_predict(net, images[j].data);
            if (net->hierarchy)
                hierarchy_predictions(p, net->outputs, net->hierarchy, 1, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if (indexes[0] == class)
            avg_acc += 1;
        for (j = 0; j < topk; ++j)
        {
            if (indexes[j] == class)
                avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
    }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    int size = net->w;
    for (i = 0; i < m; ++i)
    {
        int class = -1;
        char *path = paths[i];
        for (j = 0; j < classes; ++j)
        {
            if (strstr(path, labels[j]))
            {
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(net, resized.w, resized.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, resized.data);
        if (net->hierarchy)
            hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if (indexes[0] == class)
            avg_acc += 1;
        for (j = 0; j < topk; ++j)
        {
            if (indexes[j] == class)
                avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
    }
}

void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if (leaf_list)
        change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for (i = 0; i < m; ++i)
    {
        int class = -1;
        char *path = paths[i];
        for (j = 0; j < classes; ++j)
        {
            if (strstr(path, labels[j]))
            {
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        //grayscale_image_3c(crop);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);
        if (net->hierarchy)
            hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if (indexes[0] == class)
            avg_acc += 1;
        for (j = 0; j < topk; ++j)
        {
            if (indexes[j] == class)
                avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
    }
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *label, int top, char *imagedata, char *size)
{
    int sizeInt = atoi(size);
    //printf("value of imagedata: %.*s\n", (int)sizeof(imagedata) + 7, imagedata);
    // converting string ex: 123,242,234,234 to int array
    int totalsize = sizeInt * sizeInt * 4;
    int imgIntArray[40000] = {};

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

    char *name_list = "data/imagenet.labels.list";
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
    for (i = 0; i < 10; ++i)
    {
        int index = indexes[i];
        //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
        //else printf("%s: %f\n",names[index], predictions[index]);
        printf("%5.2f%%: %s\n", predictions[index] * 100, name_list[index]);
    }
    if (r.data != im.data)
        free_image(r);
    free_image(im);
}

void one_label_classifier(char *datacfg, char *cfgfile, char *weightfile, char *label, int top, char *imagedata, char *size)
{
    int sizeInt = atoi(size);
    //printf("value of imagedata: %.*s\n", (int)sizeof(imagedata) + 7, imagedata);
    // converting string ex: 123,242,234,234 to int array
    int totalsize = sizeInt * sizeInt * 4;
    int imgIntArray[40000] = {};

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

    char *name_list = "data/imagenet.labels.list";
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
    printf("%.8f", predictions[i]);
    if (r.data != im.data)
        free_image(r);
    free_image(im);
    //if (filename) break;
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
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *label = (argc > 6) ? argv[6] : 0;
    char *imagedata = (argc > 7) ? argv[7] : 0;
    char *size = (argc > 8) ? argv[8] : 0;
    if (0 == strcmp(argv[2], "predict"))
        predict_classifier(data, cfg, weights, label, top, imagedata, size);
    if (0 == strcmp(argv[2], "one_label"))
        one_label_classifier(data, cfg, weights, label, top, imagedata, size);
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
