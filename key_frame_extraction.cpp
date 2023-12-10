#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <cstdlib>
#include <set>
#include <omp.h>
#include <mpi.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

using namespace std;

struct VideoContext {
    AVFormatContext* format_ctx;
    AVCodecContext* codec_ctx;
    int video_stream_index;
};


int64_t FrameToPts(AVStream* pavStream, int64_t frame)
{
    return (int64_t(frame) * pavStream->r_frame_rate.den *  pavStream->time_base.den) /
           (int64_t(pavStream->r_frame_rate.num) *
            pavStream->time_base.num);
}

void print_vector(std::vector<int>& v) {
    for(int i: v) {
        std::cout<<i<<", ";
    }
    std::cout<<std::endl;
}

void print_set(std::set<int64_t>& v) {
    for(int64_t i: v) {
        std::cout<<i<<", ";
    }
    std::cout<<std::endl;
}


VideoContext get_codec_context_for_video_file(const string& input_filename) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    AVFormatContext* format_ctx = NULL;

    if (avformat_open_input(&format_ctx, input_filename.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Error opening input file." << std::endl;
        exit(1);
    }

    if (avformat_find_stream_info(format_ctx, NULL) < 0) {
        std::cerr << "Error finding stream information." << std::endl;
        exit(1);
    }

    const AVCodec* codec = nullptr;
    int video_stream_index = -1;

    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            codec = avcodec_find_decoder(format_ctx->streams[i]->codecpar->codec_id);
            if (!codec) {
                std::cerr << "Error finding codec." << std::endl;
                exit(1);
            }
            break;
        }
    }
    AVCodecContext * codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Error allocating codec context." << std::endl;
        exit(1);
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar) < 0) {
        std::cerr << "Error setting codec parameters." << std::endl;
        exit(1);
    }

    if (avcodec_open2(codec_ctx, codec_ctx->codec, nullptr) < 0) {
        std::cerr << "Error opening codec." << std::endl;
        exit(1);
    }

    VideoContext ctx{};
    ctx.codec_ctx = codec_ctx;
    ctx.format_ctx = format_ctx;
    ctx.video_stream_index = video_stream_index;
    return ctx;
}


void process_video_naive(const std::string& input_filename) {
    cout<<endl<<"Starting Naive approach"<<endl;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    VideoContext v_ctx = get_codec_context_for_video_file(input_filename);
    AVFormatContext* format_ctx = v_ctx.format_ctx;
    AVCodecContext * codec_ctx = v_ctx.codec_ctx;
    int video_stream_index = v_ctx.video_stream_index;

    // Divide the video into 4 parts
    int64_t total_frames = format_ctx->streams[video_stream_index]->nb_frames;
    int64_t total_key_frames = 0;
    std::set<int64_t> key_frame_numbers;

    AVPacket* packet = av_packet_alloc();
    int ct = 0;
    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            ct++;
            // Check if the packet contains a key frame
            if (packet->flags & AV_PKT_FLAG_KEY) {
                total_key_frames++;
                key_frame_numbers.insert(packet->pts);
            }
        }

        av_packet_unref(packet);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Total Frames: " << total_frames<<std::endl;
    std::cout << "Key Frames " << key_frame_numbers.size() << std::endl;
//    std::cout << "Key Frame Indices: ";
//    print_set(key_frame_numbers);

    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);

    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Elapsed time with Naive approach is : " << elapsed.count() << " " << std::endl;
}


void process_video_omp(const std::string& input_filename, int parallelism) {
    const int SEGMENTS = parallelism;

    cout<<endl<<"Starting OMP-"<<SEGMENTS<<endl;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    VideoContext _v_ctx = get_codec_context_for_video_file(input_filename);
    int video_stream_index = _v_ctx.video_stream_index;

    // /Users/farhan/Desktop/workspaces/study/ms/semester3/ms_project/scanner/datasets/large_videos/huge_5gb_video.mp4

    // Divide the video into segments
    int64_t total_duration = _v_ctx.format_ctx->streams[video_stream_index]->duration;
    int64_t duration_per_part = total_duration / SEGMENTS;
    int64_t frames_per_part = _v_ctx.format_ctx->streams[video_stream_index]->nb_frames / SEGMENTS;
    int64_t total_key_frames = 0;
    std::set<int64_t> key_frame_numbers;

    # pragma omp parallel num_threads(SEGMENTS)
    {
        # pragma omp for reduction(+:total_key_frames)
        for (int part = 0; part < SEGMENTS; ++part) {
            VideoContext v_ctx = get_codec_context_for_video_file(input_filename);
            AVFormatContext* format_ctx = v_ctx.format_ctx;

            // Seek to the start of the part
            int64_t segment_start_time = part * duration_per_part;
            int64_t end_time = (part == SEGMENTS-1) ? total_duration : ((part + 1) * duration_per_part - 1);

            if(av_seek_frame(format_ctx, video_stream_index, segment_start_time, AVSEEK_FLAG_BACKWARD) < 0) {
                std::cerr << "Error seeking to time : " << segment_start_time <<std::endl;
            }

            int64_t i_frame_count = 0;

            AVPacket* packet = av_packet_alloc();
            int frames_processed = 0;

            while (av_read_frame(format_ctx, packet) >= 0) {
                if (packet->stream_index == video_stream_index) {
                    if(frames_processed > frames_per_part) {
                        av_packet_unref(packet);
                        break;
                    }

                    if (packet->flags & AV_PKT_FLAG_KEY) {
                        i_frame_count++;
                        key_frame_numbers.insert(packet->pts);
                    }
                    frames_processed += 1;
                }
                av_packet_unref(packet);
            }
            total_key_frames += i_frame_count;
            avcodec_free_context(&v_ctx.codec_ctx);
            avformat_close_input(&v_ctx.format_ctx);
        }
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
    std::cout << " Total Frames: " << _v_ctx.format_ctx->streams[video_stream_index]->nb_frames<<std::endl;
    std::cout << " Key Frames " << key_frame_numbers.size() << std::endl;
//    std::cout << "Key Frame Indices: ";
//    print_set(key_frame_numbers);

    avcodec_free_context(&_v_ctx.codec_ctx);
    avformat_close_input(&_v_ctx.format_ctx);

    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Elapsed time for OMP Shared Parallel with "<<SEGMENTS<<"-way parallelism is : " << elapsed.count() << " " << std::endl;
}


void process_video_distributed(int argc, char** argv) {

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);

    int rank, size, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_size( MPI_COMM_WORLD, &nranks);

    if(rank == 0) {
        cout<<endl<<"Starting Distributed MPI"<<endl;
    }

    vector<int> key_frame_count_for_rank;
    key_frame_count_for_rank.resize(nranks);

    char* input_filename = argv[1];
    VideoContext v_ctx = get_codec_context_for_video_file(input_filename);
    int video_stream_index = v_ctx.video_stream_index;
    AVFormatContext* format_ctx = v_ctx.format_ctx;

    int64_t total_duration = v_ctx.format_ctx->streams[video_stream_index]->duration;
    int64_t  total_frames = v_ctx.format_ctx->streams[video_stream_index]->nb_frames;
    int64_t duration_per_part = total_duration / nranks;
    int64_t frames_per_part = total_frames / nranks;
    int64_t total_key_frames = 0;
    int64_t segment_start_time = rank * duration_per_part;
    std::set<int64_t> key_frame_numbers;

    if(av_seek_frame(format_ctx, video_stream_index, segment_start_time, AVSEEK_FLAG_BACKWARD) < 0) {
        std::cerr << "Error seeking to time : " << segment_start_time <<std::endl;
    }

    AVPacket* packet = av_packet_alloc();
    int frames_processed = 0;
    vector<int64_t> local_key_frame_numbers;

    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            if(frames_processed > frames_per_part) {
                av_packet_unref(packet);
                break;
            }

            if (packet->flags & AV_PKT_FLAG_KEY) {
                local_key_frame_numbers.push_back(packet->pts);
            }
            frames_processed += 1;
        }
        av_packet_unref(packet);
    }
    avcodec_free_context(&v_ctx.codec_ctx);
    avformat_close_input(&v_ctx.format_ctx);


    // all processes must complete there work before we start collecting the results
    std::flush(std::cout);
    MPI_Barrier(MPI_COMM_WORLD);

    int local_key_frame_count = local_key_frame_numbers.size();

    // Gather the count of key frames from each process
    std::flush(std::cout);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&local_key_frame_count, 1, MPI_INT, key_frame_count_for_rank.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int64_t > key_frames;

    if (rank == 0) {
        for (int i = 0; i < nranks; ++i) {
            total_key_frames += key_frame_count_for_rank[i];
        }
        key_frames.resize(total_key_frames);

    }

    // Calculate displacements for variable-sized data
    std::vector<int> displacements(size, 0);
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i - 1] + key_frame_count_for_rank[i - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Gather the modified vectors back to the master
    MPI_Gatherv(local_key_frame_numbers.data(), local_key_frame_numbers.size(), MPI_LONG_LONG, key_frames.data(), key_frame_count_for_rank.data(), displacements.data(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

    // The master combines the received vectors into a single set
    if (rank == 0) {
        std::set<int64_t> final_key_frame_numbers(key_frames.begin(), key_frames.end());
        std::cout << " Total Frames: " << total_frames <<std::endl;
        std::cout << " Key Frames " << final_key_frame_numbers.size() << std::endl;
//        std::cout << "Key Frame Indices: ";
//        print_set(final_key_frame_numbers);
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << " Elapsed time for Distributed MPI is : " << elapsed.count() << " " << std::endl;
    }

    avcodec_free_context(&v_ctx.codec_ctx);
    avformat_close_input(&v_ctx.format_ctx);

    MPI_Finalize();
}


void process_video_omp_gpu(const std::string& input_filename, int parallelism) {
    const int SEGMENTS = 100;

    cout<<endl<<"Starting OMP GPU OFFLOAD"<<endl;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    VideoContext _v_ctx = get_codec_context_for_video_file(input_filename);
    int video_stream_index = _v_ctx.video_stream_index;

    // /Users/farhan/Desktop/workspaces/study/ms/semester3/ms_project/scanner/datasets/large_videos/huge_5gb_video.mp4

    // Divide the video into segments
    int64_t total_duration = _v_ctx.format_ctx->streams[video_stream_index]->duration;
    int64_t duration_per_part = total_duration / SEGMENTS;
    int64_t frames_per_part = _v_ctx.format_ctx->streams[video_stream_index]->nb_frames / SEGMENTS;
    std::set<int64_t> key_frame_numbers;

    # pragma omp target teams distribute parallel for map(to: frames_per_part) map(tofrom: key_frame_numbers) map(to: input_filename)
    for (int part = 0; part < SEGMENTS; ++part) {
        VideoContext v_ctx = get_codec_context_for_video_file(input_filename);
        int video_stream = _v_ctx.video_stream_index;
        AVFormatContext* format_ctx = v_ctx.format_ctx;

        // Seek to the start of the part
        int64_t segment_start_time = part * duration_per_part;
        int64_t end_time = (part == SEGMENTS-1) ? total_duration : ((part + 1) * duration_per_part - 1);

        if(av_seek_frame(format_ctx, video_stream, segment_start_time, AVSEEK_FLAG_BACKWARD) < 0) {
            std::cerr << "Error seeking to time : " << segment_start_time <<std::endl;
        }

        int64_t i_frame_count = 0;

        AVPacket* packet = av_packet_alloc();
        int frames_processed = 0;

        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == video_stream) {
                if(frames_processed > frames_per_part) {
                    av_packet_unref(packet);
                    break;
                }

                if (packet->flags & AV_PKT_FLAG_KEY) {
                    i_frame_count++;

                    #pragma omp critical
                    key_frame_numbers.insert(packet->pts);
                }
                frames_processed += 1;
            }
            av_packet_unref(packet);
        }
        avcodec_free_context(&v_ctx.codec_ctx);
        avformat_close_input(&v_ctx.format_ctx);
    }

    // Access the modified vector back on the host
    #pragma omp target update from(key_frame_numbers)

    // Unmap the vector from GPU device memory
    #pragma omp target exit data map(release: key_frame_numbers)

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
    std::cout << " Total Frames: " << _v_ctx.format_ctx->streams[video_stream_index]->nb_frames<<std::endl;
    std::cout << " Key Frames " << key_frame_numbers.size() << std::endl;
//    std::cout << "Key Frame Indices: ";
//    print_set(key_frame_numbers);

    avcodec_free_context(&_v_ctx.codec_ctx);
    avformat_close_input(&_v_ctx.format_ctx);

    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Elapsed time for OMP GPU OFFLOAD with "<<SEGMENTS<<"-way parallelism is : " << elapsed.count() << " " << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file1 output_file2 output_file3 output_file4" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];

    #if DISTRIBUTED_IMPL == 1
        process_video_distributed(argc, argv);
    #else
        process_video_naive(input_filename);
        process_video_omp(input_filename, 16);
        process_video_omp(input_filename, 32);
        process_video_omp(input_filename, 64);
        process_video_omp_gpu(input_filename, 64);
    #endif

    return 0;
}
