// song_cover_page.h — one-click song-cover flow.
#pragma once

#include <QElapsedTimer>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

class AudioPlayer;
class QCheckBox;
class QComboBox;
class QDialog;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QProcess;
class QProgressBar;
class QPushButton;
class QSlider;
class QSpinBox;
class QToolButton;

// One-click song cover:
//   song -> [separate vocals/instrumental] -> [optional: split background vocals]
//        -> [optional: de-reverb main vocals] -> [convert vocals with RVC]
//        -> [optional: convert background]
//        -> [optional: pitch-shift instrumental] -> mix -> output (WAV/FLAC/MP3/OGG).
class SongCoverPage : public QWidget {
    Q_OBJECT
public:
    explicit SongCoverPage(QWidget* parent = nullptr);
    ~SongCoverPage() override;

private slots:
    void browse_song();
    void browse_output();
    void refresh_models();
    void start_run();
    void cancel_run();
    void show_log();
    void toggle_playback();
    void stop_playback();
    void on_slider_moved(int value);
    void on_slider_released();
    void on_player_position(double position, double duration);
    void on_player_state();

private:
    enum class StepKind { Separate, SplitBackup, DeReverb, ConvertMain, ConvertBackup, ShiftInstrumental, Mix };

    void build_ui();
    void populate_models();
    void log(const QString& text);
    void set_status(const QString& text);
    void set_step_state(int index, const QChar& mark);
    void complete_step();
    void fail(const QString& message);
    void finish_run(bool success);
    void start_process(const QString& program, const QStringList& args);
    void on_process_finished(int exit_code, int exit_status);
    void run_step(int index);
    void run_mix();
    bool prepare_instrumental();
    bool prepare_split();
    bool resolve_dereverb_output();
    void load_output_for_playback(const QString& path);
    bool encode_to(const QString& wav_path, const QString& target_path, QString* err);
    int total_pitch() const;
    QStringList rvc_args(const QString& in, const QString& out, bool auto_pitch) const;
    QString index_path_for(const QString& voice_model) const;

    // Inputs
    QLineEdit* song_edit_ = nullptr;
    QLineEdit* output_edit_ = nullptr;
    QComboBox* voice_combo_ = nullptr;
    QComboBox* vocal_sep_combo_ = nullptr;
    QCheckBox* flip_stems_check_ = nullptr;
    QCheckBox* split_backup_check_ = nullptr;
    QComboBox* backup_sep_combo_ = nullptr;
    QCheckBox* flip_backup_check_ = nullptr;
    QCheckBox* dereverb_check_ = nullptr;
    QComboBox* dereverb_combo_ = nullptr;
    QComboBox* extractor_combo_ = nullptr;
    QComboBox* f0_combo_ = nullptr;
    QSpinBox* octaves_spin_ = nullptr;
    QSpinBox* semitones_spin_ = nullptr;
    QComboBox* device_combo_ = nullptr;
    QDoubleSpinBox* index_rate_spin_ = nullptr;
    QDoubleSpinBox* protect_spin_ = nullptr;
    QDoubleSpinBox* rms_mix_spin_ = nullptr;
    QDoubleSpinBox* main_gain_spin_ = nullptr;
    QDoubleSpinBox* inst_gain_spin_ = nullptr;
    QDoubleSpinBox* backup_gain_spin_ = nullptr;
    QSpinBox* sid_spin_ = nullptr;
    QComboBox* output_sr_combo_ = nullptr;
    QCheckBox* proposed_pitch_check_ = nullptr;
    QDoubleSpinBox* pitch_threshold_spin_ = nullptr;

    // Output widgets
    QListWidget* steps_list_ = nullptr;
    QPlainTextEdit* log_view_ = nullptr;   // lives inside the (lazy) log dialog
    QDialog* log_dialog_ = nullptr;
    QPushButton* log_button_ = nullptr;
    QProgressBar* progress_bar_ = nullptr;
    QLabel* status_label_ = nullptr;
    QPushButton* run_button_ = nullptr;
    QPushButton* cancel_button_ = nullptr;

    // Playback
    AudioPlayer* player_ = nullptr;
    QPushButton* play_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;
    QSlider* position_slider_ = nullptr;
    QLabel* time_label_ = nullptr;
    bool slider_dragging_ = false;

    // Runtime state
    QProcess* process_ = nullptr;
    QVector<StepKind> plan_;
    int step_index_ = -1;
    bool running_ = false;
    bool cancel_requested_ = false;
    QElapsedTimer run_timer_;
    bool instrumental_shift_applied_ = false;
    QString work_dir_;
    QString tool_sep_;
    QString tool_rvc_;
    QString tool_sox_;
    QString tool_ffmpeg_;
    QString vocals_wav_;
    QString main_vocals_wav_;
    QString background_wav_;
    QString instrumental_wav_;
    QString dereverb_wav_;
    QString conv_main_wav_;
    QString conv_backup_wav_;
    QString shifted_instrumental_wav_;
};
