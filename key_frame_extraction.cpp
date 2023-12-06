#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <cstdlib>
#include <set>
#include <omp.h>

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

void print_vector(std::set<int64_t>& v) {
    for(int64_t i: v) {
        std::cout<<i<<", ";
    }
    std::cout<<std::endl;
}

VideoContext get_codec_context_for_video_file(const string& input_filename) {
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

    std::cout << "Total Frames: " << total_frames<<std::endl;
    std::cout << "Total Frames Requests: " << ct<<std::endl;
    std::cout << "Key Frames " << key_frame_numbers.size() << std::endl;
    std::cout << "Key Frame Indices: ";
    print_vector(key_frame_numbers);

    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
}

void process_video(const std::string& input_filename) {
    VideoContext v_ctx = get_codec_context_for_video_file(input_filename);
    AVFormatContext* format_ctx = v_ctx.format_ctx;
    AVCodecContext * codec_ctx = v_ctx.codec_ctx;
    int video_stream_index = v_ctx.video_stream_index;

    // Divide the video into 4 parts
    int64_t total_frames = format_ctx->streams[video_stream_index]->nb_frames;
    int64_t frames_per_part = total_frames / 4;
    int64_t total_key_frames = 0;
    std::set<int64_t> key_frame_numbers;

    for (int part = 0; part < 4; ++part) {
        // Seek to the start of the part
        int64_t start_frame = part * frames_per_part;
        int64_t curr_frame = start_frame;

        int64_t tmp_stmp = FrameToPts(format_ctx->streams[video_stream_index], start_frame);
        if(av_seek_frame(format_ctx, video_stream_index, tmp_stmp, 0) < 0) {
            std::cerr << "Error seeking to frame : " << start_frame <<std::endl;
        }

        // Process frames in the part
        int64_t end_frame = (part == 3) ? total_frames : ((part + 1) * frames_per_part - 1);
        int64_t i_frame_count = 0;

        AVPacket* packet = av_packet_alloc();

        while (av_read_frame(format_ctx, packet) >= 0 && curr_frame <= end_frame) {
            if (packet->stream_index == video_stream_index) {
                AVFrame* frame = av_frame_alloc();
                if (!frame) {
                    std::cerr << "Error allocating frame." << std::endl;
                    break;
                }

                int ret = EAGAIN;
                while(abs(ret) == EAGAIN) {
                    avcodec_send_packet(codec_ctx, packet);
                    ret = avcodec_receive_frame(codec_ctx, frame);
                }

                if(ret < 0 ) {
                    std::cerr << "Error decoding frame." << std::endl;
                }
                else {
                    curr_frame += 1;
                    if (frame->pict_type == AV_PICTURE_TYPE_I) {
                        // Count I-frames
                        ++i_frame_count;
                        key_frame_numbers.insert(frame->pts);
                    }
                }

                av_frame_free(&frame);
            }

            av_packet_unref(packet);
        }
        total_key_frames += i_frame_count;
    }

    std::cout << "Total Frames: " << total_frames<<std::endl;
    std::cout << "Key Frames " << key_frame_numbers.size() << std::endl;
    std::cout << "Key Frame Indices: ";
    print_vector(key_frame_numbers);

    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
}

void process_video_omp(const std::string& input_filename) {
    VideoContext _v_ctx = get_codec_context_for_video_file(input_filename);
    int video_stream_index = _v_ctx.video_stream_index;

    // Divide the video into segments
    int64_t total_frames = _v_ctx.format_ctx->streams[video_stream_index]->nb_frames;
    int64_t frames_per_part = total_frames / 4;
    int64_t total_key_frames = 0;
    std::set<int64_t> key_frame_numbers;

    # pragma omp parallel for reduction(+:total_key_frames)
    for (int part = 0; part < 4; ++part) {
        VideoContext v_ctx = get_codec_context_for_video_file(input_filename);
        AVFormatContext* format_ctx = v_ctx.format_ctx;
        AVCodecContext * codec_ctx = v_ctx.codec_ctx;
        // Seek to the start of the part
        int64_t start_frame = part * frames_per_part;
        int64_t curr_frame = start_frame;

        int64_t tmp_stmp = FrameToPts(format_ctx->streams[video_stream_index], start_frame);
        if(av_seek_frame(format_ctx, video_stream_index, tmp_stmp, 0) < 0) {
            std::cerr << "Error seeking to frame : " << start_frame <<std::endl;
        }

        // Process frames in the part
        int64_t end_frame = (part == 3) ? total_frames : ((part + 1) * frames_per_part - 1);
        int64_t i_frame_count = 0;

        AVPacket* packet = av_packet_alloc();

        while (av_read_frame(format_ctx, packet) >= 0 && curr_frame <= end_frame) {
            if (packet->stream_index == video_stream_index) {
                AVFrame* frame = av_frame_alloc();
                if (!frame) {
                    std::cerr << "Error allocating frame." << std::endl;
                    break;
                }

                int ret = EAGAIN;
                while(abs(ret) == EAGAIN) {
                    avcodec_send_packet(codec_ctx, packet);
                    ret = avcodec_receive_frame(codec_ctx, frame);
                }

                if(ret < 0 ) {
                    std::cerr << "Error decoding frame." << std::endl;
                }
                else {
                    curr_frame += 1;
                    if (frame->pict_type == AV_PICTURE_TYPE_I) {
                        // Count I-frames
                        ++i_frame_count;
                        key_frame_numbers.insert(frame->pts);
                    }
                }

                av_frame_free(&frame);
            }

            av_packet_unref(packet);
        }
        total_key_frames += i_frame_count;

        avcodec_free_context(&v_ctx.codec_ctx);
        avformat_close_input(&v_ctx.format_ctx);
    }

    std::cout << "Total Frames: " << total_frames<<std::endl;
    std::cout << "Key Frames " << key_frame_numbers.size() << std::endl;
    std::cout << "Key Frame Indices: ";
    print_vector(key_frame_numbers);

    avcodec_free_context(&_v_ctx.codec_ctx);
    avformat_close_input(&_v_ctx.format_ctx);
}

void process_video_omp1(const std::string& input_filename) {
    VideoContext _v_ctx = get_codec_context_for_video_file(input_filename);
    int video_stream_index = _v_ctx.video_stream_index;

    // /Users/farhan/Desktop/workspaces/study/ms/semester3/ms_project/scanner/datasets/large_videos/huge_5gb_video.mp4

    // Divide the video into segments
    const int SEGMENTS = 64;
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

    std::cout << "Total Frames: " << _v_ctx.format_ctx->streams[video_stream_index]->nb_frames<<std::endl;
    std::cout << "Key Frames " << key_frame_numbers.size() << std::endl;
    std::cout << "Key Frame Indices: ";
    print_vector(key_frame_numbers);

    avcodec_free_context(&_v_ctx.codec_ctx);
    avformat_close_input(&_v_ctx.format_ctx);
}

void process_video_opt(const std::string& input_filename) {

    AVFormatContext* format_ctx = NULL;

    if (avformat_open_input(&format_ctx, input_filename.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Error opening input file." << std::endl;
        return;
    }

    if (avformat_find_stream_info(format_ctx, NULL) < 0) {
        std::cerr << "Error finding stream information." << std::endl;
        return;
    }

    const AVCodec* codec = nullptr;
    int video_stream_index = -1;

    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            codec = avcodec_find_decoder(format_ctx->streams[i]->codecpar->codec_id);
            if (!codec) {
                std::cerr << "Error finding codec." << std::endl;
                return;
            }
            break;
        }
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Error allocating codec context." << std::endl;
        return;
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar) < 0) {
        std::cerr << "Error setting codec parameters." << std::endl;
        return;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Error opening codec." << std::endl;
        return;
    }

    // assuming there will be at least 1 key frame
    int total_key_frames = 1;
    int64_t last_key_frame_time = 0;

    while(true) {
        // seek to the next key frame
        if(av_seek_frame(format_ctx, -1, 170, 0) < 0) {
            std::cerr << "Error seeking to frame : " << last_key_frame_time <<std::endl;
            break;
        }

        AVPacket* packet = av_packet_alloc();

        av_read_frame(format_ctx, packet);
        AVFrame* frame = av_frame_alloc();
        int ret = EAGAIN;
        while(abs(ret) == EAGAIN) {
            avcodec_send_packet(codec_ctx, packet);
            ret = avcodec_receive_frame(codec_ctx, frame);
        }
        if(ret < 0 ) {
            std::cerr << "Error decoding frame." << std::endl;
            break;
        }
        else if(frame->key_frame) {
            total_key_frames += 1;
            last_key_frame_time = frame->pts;
        }
    }


    // Divide the video into 4 parts
    int64_t total_frames = format_ctx->streams[video_stream_index]->nb_frames;

    std::cout << "Total Frames: " << total_frames<<std::endl;
    std::cout << "Key Frames " << total_key_frames << std::endl;

    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
}

void process_video_distributed(const std::string& input_filename) {

    AVFormatContext* format_ctx = NULL;

    if (avformat_open_input(&format_ctx, input_filename.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Error opening input file." << std::endl;
        return;
    }

    if (avformat_find_stream_info(format_ctx, NULL) < 0) {
        std::cerr << "Error finding stream information." << std::endl;
        return;
    }

    const AVCodec* codec = nullptr;
    int video_stream_index = -1;

    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            codec = avcodec_find_decoder(format_ctx->streams[i]->codecpar->codec_id);
            if (!codec) {
                std::cerr << "Error finding codec." << std::endl;
                return;
            }
            break;
        }
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Error allocating codec context." << std::endl;
        return;
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar) < 0) {
        std::cerr << "Error setting codec parameters." << std::endl;
        return;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Error opening codec." << std::endl;
        return;
    }

    // assuming there will be at least 1 key frame
    int total_key_frames = 1;
    int64_t last_key_frame_time = 0;

    while(true) {
        // seek to the next key frame
        if(av_seek_frame(format_ctx, -1, 170, 0) < 0) {
            std::cerr << "Error seeking to frame : " << last_key_frame_time <<std::endl;
            break;
        }

        AVPacket* packet = av_packet_alloc();

        av_read_frame(format_ctx, packet);
        AVFrame* frame = av_frame_alloc();
        int ret = EAGAIN;
        while(abs(ret) == EAGAIN) {
            avcodec_send_packet(codec_ctx, packet);
            ret = avcodec_receive_frame(codec_ctx, frame);
        }
        if(ret < 0 ) {
            std::cerr << "Error decoding frame." << std::endl;
            break;
        }
        else if(frame->key_frame) {
            total_key_frames += 1;
            last_key_frame_time = frame->pts;
        }
    }


    // Divide the video into 4 parts
    int64_t total_frames = format_ctx->streams[video_stream_index]->nb_frames;

    std::cout << "Total Frames: " << total_frames<<std::endl;
    std::cout << "Key Frames " << total_key_frames << std::endl;

    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file1 output_file2 output_file3 output_file4" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
//    process_video_naive(input_filename);
//    process_video(input_filename);
//    process_video_omp(input_filename);
    process_video_omp1(input_filename);
//    process_video_opt(input_filename);

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Elapsed time is : " << elapsed.count() << " " << std::endl;

    return 0;
}
