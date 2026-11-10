// audio_player.h — miniaudio-based playback for rvc-ui outputs.
#pragma once

#include <QObject>
#include <QString>
#include <QTimer>

#include <memory>

// Simple play/pause/stop/seek player for generated output files
// (WAV/FLAC/MP3/OGG, decoded by miniaudio). Emits position updates on a timer
// so the UI can drive its progress bar (a future waveform widget can consume
// the same signal).
class AudioPlayer : public QObject {
    Q_OBJECT
public:
    explicit AudioPlayer(QObject* parent = nullptr);
    ~AudioPlayer() override;

    bool available() const;       // false when no audio device could be opened
    bool load(const QString& path);
    bool loaded() const;
    bool playing() const;

    double position_seconds() const;
    double duration_seconds() const;

public slots:
    void play();
    void pause();
    void stop();
    void toggle_play();
    void seek_seconds(double seconds);

signals:
    void position_changed(double position, double duration);
    void state_changed();

private:
    void tick();

    struct Impl;
    std::unique_ptr<Impl> impl_;
    QTimer timer_;
};
