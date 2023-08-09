#include <iostream>
#include <vector>
#include "lodepng.h"
#include <fstream>
#include <mpi.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>

// Estructura para representar una imagen
struct Image {
    std::vector<unsigned char> pixels;
    unsigned width;
    unsigned height;
};

// Función para cargar una imagen PGM en una matriz de píxeles
bool loadPGM(const std::string &filename, Image &image) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_red), "Error al abrir el archivo: {}", filename);
        return false;
    }

    // Leer el encabezado de la imagen PGM
    std::string formato;
    file >> formato;
    file >> image.width >> image.height;

    int max_value;
    file >> max_value;

    // Verificar el tipo de formato PGM y convertir de P5 a P2 si es necesario
    if (formato == "P5") {
        // Leer los píxeles de la imagen en formato binario (P5)
        image.pixels.clear();
        image.pixels.resize(image.width * image.height);

        file.ignore(); // Ignorar el salto de línea después del valor máximo

        for (unsigned i = 0; i < image.width * image.height; i++) {
            unsigned char pixel_value;
            file.read(reinterpret_cast<char *>(&pixel_value), 1);
            image.pixels[i] = pixel_value;
        }
    } else if (formato == "P2") {
        // Leer los píxeles de la imagen en formato de texto (P2)
        image.pixels.clear();
        image.pixels.resize(image.width * image.height);

        for (unsigned i = 0; i < image.width * image.height; i++) {
            int pixel_value;
            file >> pixel_value;
            image.pixels[i] = static_cast<unsigned char>(pixel_value);
        }
    } else {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_red), "Formato de imagen no soportado: {} \n", formato);
        return false;
    }

    file.close();
    return true;
}


// Función para cargar una imagen PNG en una matriz de píxeles
bool loadPNG(const std::string &filename, Image &image) {
    std::vector<unsigned char> png_image; // Vector para almacenar los bytes de la imagen PNG
    unsigned error = lodepng::decode(png_image, image.width, image.height, filename);

    if (error) {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_red), "Error al cargar la imagen: {} \n",
                   lodepng_error_text(error));
        return false;
    }

    // Convertir los bytes de la imagen a escala de grises (promedio de los componentes RGB)
    image.pixels.clear();
    image.pixels.resize(image.width * image.height);

    for (size_t i = 0; i < png_image.size(); i += 4) {
        unsigned char intensity = (png_image[i] + png_image[i + 1] + png_image[i + 2]) / 3;
        image.pixels[i / 4] = intensity;
    }

    return true;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Cargar las imágenes en el proceso 0
    Image imagePNG, imagePGM;

    if (rank == 0) {
        std::string filenamePNG = "/workspace/lab02-1/img.png"; // Ruta de la imagen PNG
        std::string filenamePGM = "/workspace/lab02-1/ejem.pgm";

        if (!loadPNG(filenamePNG, imagePNG)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!loadPGM(filenamePGM, imagePGM)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int num_bins = 20; // Número de bins para el histograma
    int intensity_min = 0;
    int intensity_max = 255;

    // Broadcast de los componentes de las imágenes desde el proceso 0 a los demás procesos
    MPI_Bcast(&imagePNG.width, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imagePNG.height, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    imagePNG.pixels.resize(imagePNG.width * imagePNG.height);
    MPI_Bcast(&imagePNG.pixels[0], imagePNG.width * imagePNG.height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Bcast(&imagePGM.width, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imagePGM.height, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    imagePGM.pixels.resize(imagePGM.width * imagePGM.height);
    MPI_Bcast(&imagePGM.pixels[0], imagePGM.width * imagePGM.height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    {
        // Divide el trabajo entre los procesos
        int local_chunk_size = imagePGM.pixels.size() / size;
        int local_start = rank * local_chunk_size;
        int local_end = (rank == size - 1) ? imagePGM.pixels.size() : local_start + local_chunk_size;
        int bin_size = (intensity_max - intensity_min + num_bins) / num_bins;

        std::vector<int> local_histogram(num_bins, 0);

        for (int i = local_start; i < local_end; i++) {
            int bin_index = (imagePGM.pixels[i] - intensity_min) / bin_size;
            local_histogram[bin_index]++;
        }

        // Reducción de los histogramas locales en el histograma global usando MPI_Reduce
        std::vector<int> global_histogram(num_bins, 0);
        MPI_Reduce(&local_histogram[0], &global_histogram[0], num_bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // El proceso 0 tiene el histograma final en global_histogram
        if (rank == 0) {
            int total_sum = 0;
            for (int count: global_histogram) {
                total_sum += count;
            }
            fmt::print(fmt::emphasis::reverse | fg(fmt::color::gray), "Imagen PGM \n");
            for (int i = 0; i < num_bins; i++) {
                int bin_start = i * (256 / num_bins);
                int bin_end = (i == num_bins - 1) ? 255 : (i + 1) * (256 / num_bins) - 1;
                double porcentaje = static_cast<double>(global_histogram[i]) / total_sum;
                fmt::print(fmt::emphasis::blink | fg(fmt::color::chartreuse), "{} - {}: {} ({:0.6} %)\n", bin_start,
                           bin_end,
                           global_histogram[i], porcentaje * 100.0);
            }
        }
    }

    {
        // Divide el trabajo entre los procesos
        int local_chunk_size = imagePNG.pixels.size() / size;
        int local_start = rank * local_chunk_size;
        int local_end = (rank == size - 1) ? imagePNG.pixels.size() : local_start + local_chunk_size;
        int bin_size = (intensity_max - intensity_min + num_bins) / num_bins;

        std::vector<int> local_histogram(num_bins, 0);

        for (int i = local_start; i < local_end; i++) {
            int bin_index = (imagePNG.pixels[i] - intensity_min) / bin_size;
            local_histogram[bin_index]++;
        }

        // Reducción de los histogramas locales en el histograma global usando MPI_Reduce
        std::vector<int> global_histogram(num_bins, 0);
        MPI_Reduce(&local_histogram[0], &global_histogram[0], num_bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // El proceso 0 tiene el histograma final en global_histogram
        if (rank == 0) {
            int total_sum = 0;
            for (int count: global_histogram) {
                total_sum += count;
            }
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "*************************************************** \n");
            fmt::print(fmt::emphasis::reverse | fg(fmt::color::gray), "Imagen PNG \n");
            for (int i = 0; i < num_bins; i++) {
                int bin_start = i * (256 / num_bins);
                int bin_end = (i == num_bins - 1) ? 255 : (i + 1) * (256 / num_bins) - 1;
                double porcentaje = static_cast<double>(global_histogram[i]) / total_sum;
                fmt::print(fmt::emphasis::blink | fg(fmt::color::chartreuse), "{} - {}: {} ({:0.6} %)\n", bin_start,
                           bin_end,
                           global_histogram[i], porcentaje * 100.0);
            }
        }
    }

    MPI_Finalize();
    return 0;
}