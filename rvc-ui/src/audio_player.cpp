// audio_player.cpp — miniaudio-based playback for rvc-ui outputs.
//
// Uses the high-level ma_engine/ma_sound API from the miniaudio implementation
// compiled in miniaudio_impl.cpp. Playback streams from disk and supports every
// format the decoder does (WAV/FLAC/MP3/Ogg Vorbis).
#include "audio_player.h"

#include "miniaudio.h"

#include <algorithm>

struct AudioPlayer::Impl {
    ma_engine engine;
    ma_sound sound;
    bool engine_ready = false;
    bool sound_loaded = false;
};

AudioPlayer::AudioPlayer(QObject* parent) : QObject(parent), impl_(new Impl) {
    impl_->engine_ready = (ma_engine_init(nullptr, &impl_->engine) == MA_SUCCESS);
    timer_.setInterval(100);
    connect(&timer_, &QTimer::timeout, this, &AudioPlayer::tick);
}

AudioPlayer::~AudioPlayer() {
    timer_.stop();
    if (impl_->sound_loaded) {
        ma_sound_uninit(&impl_->sound);
    }
    if (impl_->engine_ready) {
        ma_engine_uninit(&impl_->engine);
    }
}

bool AudioPlayer::available() const {
    return impl_->engine_ready;
}

bool AudioPlayer::loaded() const {
    return impl_->sound_loaded;
}

bool AudioPlayer::load(const QString& path) {
    stop();
    if (!impl_->engine_ready) {
        return false;
    }
    if (impl_->sound_loaded) {
        ma_sound_uninit(&impl_->sound);
        impl_->sound_loaded = false;
    }
    const ma_result result = ma_sound_init_from_file(
        &impl_->engine, path.toUtf8().constData(), MA_SOUND_FLAG_STREAM,
        nullptr, nullptr, &impl_->sound);
    if (result != MA_SUCCESS) {
        return false;
    }
    impl_->sound_loaded = true;
    emit position_changed(0.0, duration_seconds());
    emit state_changed();
    return true;
}

bool AudioPlayer::playing() const {
    return impl_->sound_loaded && ma_sound_is_playing(&impl_->sound) != MA_FALSE;
}

double AudioPlayer::position_seconds() const {
    if (!impl_->sound_loaded) {
        return 0.0;
    }
    float pos = 0.0f;
    if (ma_sound_get_cursor_in_seconds(&impl_->sound, &pos) != MA_SUCCESS) {
        return 0.0;
    }
    return static_cast<double>(pos);
}

double AudioPlayer::duration_seconds() const {
    if (!impl_->sound_loaded) {
        return 0.0;
    }
    float len = 0.0f;
    if (ma_sound_get_length_in_seconds(&impl_->sound, &len) != MA_SUCCESS) {
        return 0.0;
    }
    return static_cast<double>(len);
}

void AudioPlayer::play() {
    if (!impl_->sound_loaded) {
        return;
    }
    if (ma_sound_at_end(&impl_->sound) != MA_FALSE) {
        ma_sound_seek_to_pcm_frame(&impl_->sound, 0);
    }
    ma_sound_start(&impl_->sound);
    timer_.start();
    emit state_changed();
}

void AudioPlayer::pause() {
    if (!impl_->sound_loaded) {
        return;
    }
    ma_sound_stop(&impl_->sound);
    timer_.stop();
    emit state_changed();
}

void AudioPlayer::stop() {
    if (!impl_->sound_loaded) {
        return;
    }
    ma_sound_stop(&impl_->sound);
    ma_sound_seek_to_pcm_frame(&impl_->sound, 0);
    timer_.stop();
    emit position_changed(0.0, duration_seconds());
    emit state_changed();
}

void AudioPlayer::toggle_play() {
    if (playing()) {
        pause();
    } else {
        play();
    }
}

void AudioPlayer::seek_seconds(double seconds) {
    if (!impl_->sound_loaded) {
        return;
    }
    const double duration = duration_seconds();
    const double clamped = std::clamp(seconds, 0.0, duration > 0.0 ? duration : seconds);
    ma_sound_seek_to_second(&impl_->sound, static_cast<float>(clamped));
    emit position_changed(clamped, duration);
}

void AudioPlayer::tick() {
    if (!impl_->sound_loaded) {
        timer_.stop();
        return;
    }
    if (ma_sound_at_end(&impl_->sound) != MA_FALSE) {
        ma_sound_stop(&impl_->sound);
        ma_sound_seek_to_pcm_frame(&impl_->sound, 0);
        timer_.stop();
        emit position_changed(0.0, duration_seconds());
        emit state_changed();
        return;
    }
    emit position_changed(position_seconds(), duration_seconds());
}
