// song_cover_page.cpp — one-click song-cover flow.
//
// Pipeline (each external step runs rvc-sep / RVC-GGML / sox as a subprocess):
//   1. rvc-sep <vocal-separator>  song.wav    -> vocals (+ instrumental)
//   2. optional: rvc-sep <backup-separator>   vocals -> main + background stems
//   3. optional: rvc-sep <de-reverb model>    main vocals -> dry vocals
//   4. RVC-GGML conversion of the main vocals (and optionally the background)
//   5. optional: sox/ffmpeg pitch-shift of the instrumental (same amount)
//   6. in-process mix (background + main + instrumental gains) -> output WAV

#include "song_cover_page.h"

#include "audio_player.h"
#include "audio_utils.h"
#include "paths.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStandardPaths>
#include <QToolButton>
#include <QVBoxLayout>

namespace {

QString display_name(const QString& path) {
    return QFileInfo(path).fileName();
}

QString format_time_seconds(double seconds) {
    const int total = static_cast<int>(seconds + 0.5);
    return QStringLiteral("%1:%2").arg(total / 60).arg(total % 60, 2, 10, QLatin1Char('0'));
}

} // namespace

SongCoverPage::SongCoverPage(QWidget* parent) : QWidget(parent) {
    build_ui();
    populate_models();

    player_ = new AudioPlayer(this);
    connect(player_, &AudioPlayer::position_changed, this, &SongCoverPage::on_player_position);
    connect(player_, &AudioPlayer::state_changed, this, &SongCoverPage::on_player_state);
    if (!player_->available()) {
        log(QStringLiteral("Audio playback is unavailable (no audio device); covers can still be generated."));
    }
}

SongCoverPage::~SongCoverPage() {
    if (process_ != nullptr) {
        process_->kill();
        process_->waitForFinished(2000);
    }
}

void SongCoverPage::build_ui() {
    auto* layout = new QVBoxLayout(this);

    // ---------------- Source / target -------------------------------------
    auto* io_group = new QGroupBox(QStringLiteral("Song and voice"), this);
    auto* io_form = new QFormLayout(io_group);

    {
        auto* row = new QWidget(io_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        song_edit_ = new QLineEdit(row);
        song_edit_->setPlaceholderText(QStringLiteral("path to a song file (WAV/FLAC/MP3/OGG)"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, &SongCoverPage::browse_song);
        h->addWidget(song_edit_, 1);
        h->addWidget(browse);
        io_form->addRow(QStringLiteral("Song"), row);
    }
    {
        auto* row = new QWidget(io_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        voice_combo_ = new QComboBox(row);
        auto* refresh = new QPushButton(QStringLiteral("Refresh"), row);
        connect(refresh, &QPushButton::clicked, this, &SongCoverPage::refresh_models);
        h->addWidget(voice_combo_, 1);
        h->addWidget(refresh);
        io_form->addRow(QStringLiteral("Voice model"), row);
    }
    {
        auto* row = new QWidget(io_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        output_edit_ = new QLineEdit(row);
        output_edit_->setPlaceholderText(QStringLiteral("output WAV file"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, &SongCoverPage::browse_output);
        h->addWidget(output_edit_, 1);
        h->addWidget(browse);
        io_form->addRow(QStringLiteral("Output"), row);
    }
    layout->addWidget(io_group);

    // ---------------- Models ----------------------------------------------
    auto* models_group = new QGroupBox(QStringLiteral("Models"), this);
    auto* models_form = new QFormLayout(models_group);

    {
        auto* row = new QWidget(models_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        vocal_sep_combo_ = new QComboBox(row);
        flip_stems_check_ = new QCheckBox(QStringLiteral("Flip stems"), row);
        flip_stems_check_->setToolTip(QStringLiteral(
            "Some separators swap the stem order (stem 0 = instrumental, stem 1 = vocals)."));
        h->addWidget(vocal_sep_combo_, 1);
        h->addWidget(flip_stems_check_);
        models_form->addRow(QStringLiteral("Vocal separator"), row);
    }

    {
        auto* row = new QWidget(models_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        split_backup_check_ = new QCheckBox(QStringLiteral("Separate background singers"), row);
        backup_sep_combo_ = new QComboBox(row);
        backup_sep_combo_->setEnabled(false);
        flip_backup_check_ = new QCheckBox(QStringLiteral("Flip stems"), row);
        flip_backup_check_->setEnabled(false);
        flip_backup_check_->setToolTip(QStringLiteral(
            "Some splitters swap the stem order (stem 0 = background, stem 1 = main vocals)."));
        connect(split_backup_check_, &QCheckBox::toggled, backup_sep_combo_, &QComboBox::setEnabled);
        connect(split_backup_check_, &QCheckBox::toggled, flip_backup_check_, &QCheckBox::setEnabled);
        h->addWidget(split_backup_check_);
        h->addWidget(backup_sep_combo_, 1);
        h->addWidget(flip_backup_check_);
        models_form->addRow(QStringLiteral("Background"), row);
    }

    {
        auto* row = new QWidget(models_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        dereverb_check_ = new QCheckBox(QStringLiteral("De-reverb vocals"), row);
        dereverb_check_->setToolTip(QStringLiteral(
            "Optional roformer de-reverb step on the main vocals, after separation and "
            "before the RVC conversion."));
        dereverb_combo_ = new QComboBox(row);
        dereverb_combo_->setEnabled(false);
        connect(dereverb_check_, &QCheckBox::toggled, this, [this](bool on) {
            dereverb_combo_->setEnabled(on && dereverb_combo_->count() > 0);
        });
        h->addWidget(dereverb_check_);
        h->addWidget(dereverb_combo_, 1);
        models_form->addRow(QStringLiteral("De-reverb"), row);
    }

    extractor_combo_ = new QComboBox(models_group);
    models_form->addRow(QStringLiteral("Feature extractor"), extractor_combo_);
    f0_combo_ = new QComboBox(models_group);
    models_form->addRow(QStringLiteral("F0 extractor"), f0_combo_);

    layout->addWidget(models_group);

    // ---------------- Options ----------------------------------------------
    auto* options_group = new QGroupBox(QStringLiteral("Options"), this);
    auto* options_form = new QFormLayout(options_group);

    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        octaves_spin_ = new QSpinBox(row);
        octaves_spin_->setRange(-2, 2);
        octaves_spin_->setSuffix(QStringLiteral(" octaves"));
        semitones_spin_ = new QSpinBox(row);
        semitones_spin_->setRange(-12, 12);
        semitones_spin_->setSuffix(QStringLiteral(" semitones"));
        h->addWidget(octaves_spin_);
        h->addWidget(semitones_spin_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("Pitch shift"), row);
    }
    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        main_gain_spin_ = new QDoubleSpinBox(row);
        main_gain_spin_->setRange(-40.0, 20.0);
        main_gain_spin_->setSuffix(QStringLiteral(" dB"));
        inst_gain_spin_ = new QDoubleSpinBox(row);
        inst_gain_spin_->setRange(-40.0, 20.0);
        inst_gain_spin_->setSuffix(QStringLiteral(" dB"));
        backup_gain_spin_ = new QDoubleSpinBox(row);
        backup_gain_spin_->setRange(-40.0, 20.0);
        backup_gain_spin_->setSuffix(QStringLiteral(" dB"));
        h->addWidget(new QLabel(QStringLiteral("main"), row));
        h->addWidget(main_gain_spin_);
        h->addWidget(new QLabel(QStringLiteral("inst"), row));
        h->addWidget(inst_gain_spin_);
        h->addWidget(new QLabel(QStringLiteral("background"), row));
        h->addWidget(backup_gain_spin_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("Mix gains"), row);
    }
    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        index_rate_spin_ = new QDoubleSpinBox(row);
        index_rate_spin_->setRange(0.0, 1.0);
        index_rate_spin_->setSingleStep(0.05);
        index_rate_spin_->setValue(0.75);
        protect_spin_ = new QDoubleSpinBox(row);
        protect_spin_->setRange(0.0, 0.5);
        protect_spin_->setSingleStep(0.01);
        protect_spin_->setValue(0.0);
        h->addWidget(new QLabel(QStringLiteral("index rate"), row));
        h->addWidget(index_rate_spin_);
        h->addWidget(new QLabel(QStringLiteral("protect"), row));
        h->addWidget(protect_spin_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("RVC"), row);
    }
    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        proposed_pitch_check_ = new QCheckBox(QStringLiteral("Auto pitch (proposed)"), row);
        pitch_threshold_spin_ = new QDoubleSpinBox(row);
        pitch_threshold_spin_->setRange(50.0, 500.0);
        pitch_threshold_spin_->setValue(255.0);
        pitch_threshold_spin_->setSuffix(QStringLiteral(" Hz"));
        pitch_threshold_spin_->setEnabled(false);
        pitch_threshold_spin_->setToolTip(QStringLiteral(
            "Target median f0 for the converted voice: 155 Hz (male) / 255 Hz (female); "
            "shift clamped to +/-12 semitones. Background and instrumental keep the manual shift only."));
        connect(proposed_pitch_check_, &QCheckBox::toggled,
                pitch_threshold_spin_, &QDoubleSpinBox::setEnabled);
        h->addWidget(proposed_pitch_check_);
        h->addWidget(pitch_threshold_spin_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("Pitch"), row);
    }
    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        rms_mix_spin_ = new QDoubleSpinBox(row);
        rms_mix_spin_->setRange(0.0, 1.0);
        rms_mix_spin_->setSingleStep(0.05);
        rms_mix_spin_->setValue(0.0);
        sid_spin_ = new QSpinBox(row);
        sid_spin_->setRange(0, 500);
        device_combo_ = new QComboBox(row);
        device_combo_->addItem(QStringLiteral("GPU (Vulkan)"), 0);
        device_combo_->addItem(QStringLiteral("CPU"), 99);
        output_sr_combo_ = new QComboBox(row);
        output_sr_combo_->addItem(QStringLiteral("32000 Hz"), 32000);
        output_sr_combo_->addItem(QStringLiteral("40000 Hz"), 40000);
        output_sr_combo_->addItem(QStringLiteral("44100 Hz"), 44100);
        output_sr_combo_->addItem(QStringLiteral("48000 Hz"), 48000);
        output_sr_combo_->setCurrentIndex(2);
        h->addWidget(new QLabel(QStringLiteral("rms mix"), row));
        h->addWidget(rms_mix_spin_);
        h->addWidget(new QLabel(QStringLiteral("speaker"), row));
        h->addWidget(sid_spin_);
        h->addWidget(new QLabel(QStringLiteral("device"), row));
        h->addWidget(device_combo_);
        h->addWidget(new QLabel(QStringLiteral("output rate"), row));
        h->addWidget(output_sr_combo_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("More"), row);
    }
    layout->addWidget(options_group);

    // ---------------- Run ---------------------------------------------------
    auto* run_group = new QGroupBox(QStringLiteral("Run"), this);
    auto* run_layout = new QVBoxLayout(run_group);

    steps_list_ = new QListWidget(run_group);
    steps_list_->setMinimumHeight(110);
    run_layout->addWidget(steps_list_);

    // Playback row (a seekable progress slider for now; a waveform widget can
    // replace/overlay it later).
    {
        auto* row = new QWidget(run_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        play_button_ = new QPushButton(QStringLiteral("Play"), row);
        stop_button_ = new QPushButton(QStringLiteral("Stop"), row);
        position_slider_ = new QSlider(Qt::Horizontal, row);
        position_slider_->setRange(0, 1000);
        time_label_ = new QLabel(QStringLiteral("--:-- / --:--"), row);
        play_button_->setEnabled(false);
        stop_button_->setEnabled(false);
        position_slider_->setEnabled(false);
        connect(play_button_, &QPushButton::clicked, this, &SongCoverPage::toggle_playback);
        connect(stop_button_, &QPushButton::clicked, this, &SongCoverPage::stop_playback);
        connect(position_slider_, &QSlider::sliderPressed, this, [this] { slider_dragging_ = true; });
        connect(position_slider_, &QSlider::sliderMoved, this, &SongCoverPage::on_slider_moved);
        connect(position_slider_, &QSlider::sliderReleased, this, &SongCoverPage::on_slider_released);
        h->addWidget(play_button_);
        h->addWidget(stop_button_);
        h->addWidget(position_slider_, 1);
        h->addWidget(time_label_);
        run_layout->addWidget(row);
    }

    // Progress + status row (replaces the inline log; the Log button pops the
    // full log out into a separate window).
    {
        auto* row = new QWidget(run_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        progress_bar_ = new QProgressBar(row);
        progress_bar_->setRange(0, 100);
        progress_bar_->setValue(0);
        status_label_ = new QLabel(QStringLiteral("Idle"), row);
        h->addWidget(progress_bar_, 1);
        h->addWidget(status_label_);
        run_layout->addWidget(row);
    }

    // The log view is kept for the pop-out window (created on first use).
    log_view_ = new QPlainTextEdit();
    log_view_->setReadOnly(true);

    {
        auto* row = new QWidget(run_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        log_button_ = new QPushButton(QStringLiteral("Log"), row);
        connect(log_button_, &QPushButton::clicked, this, &SongCoverPage::show_log);
        run_button_ = new QPushButton(QStringLiteral("Generate song cover"), row);
        run_button_->setDefault(true);
        cancel_button_ = new QPushButton(QStringLiteral("Cancel"), row);
        cancel_button_->setEnabled(false);
        connect(run_button_, &QPushButton::clicked, this, &SongCoverPage::start_run);
        connect(cancel_button_, &QPushButton::clicked, this, &SongCoverPage::cancel_run);
        h->addWidget(log_button_);
        h->addStretch(1);
        h->addWidget(run_button_);
        h->addWidget(cancel_button_);
        run_layout->addWidget(row);
    }
    layout->addWidget(run_group, 1);
}

void SongCoverPage::populate_models() {
    const QString root = paths::root();
    const QString models = paths::models_root();

    auto fill = [](QComboBox* combo, const QStringList& files, const QString& preferred) {
        combo->clear();
        for (const QString& f : files) {
            combo->addItem(display_name(f), f);
        }
        const QString pick = paths::preferred(files, preferred);
        if (!pick.isEmpty()) {
            const int idx = combo->findData(pick);
            if (idx >= 0) {
                combo->setCurrentIndex(idx);
            }
        }
        combo->setEnabled(combo->count() > 0);
    };

    fill(vocal_sep_combo_, paths::gguf_files(models + QStringLiteral("/roformers/vocal_separator")), QStringLiteral("FV4"));
    fill(backup_sep_combo_, paths::gguf_files(models + QStringLiteral("/roformers/backup_separator")), QStringLiteral("Male-Female"));
    fill(dereverb_combo_, paths::gguf_files(models + QStringLiteral("/roformers/de-reverb")), QStringLiteral("De-Reverb"));
    dereverb_combo_->setEnabled(dereverb_check_->isChecked() && dereverb_combo_->count() > 0);
    fill(extractor_combo_, paths::gguf_files(models + QStringLiteral("/extractors")), QStringLiteral("contentvec"));
    fill(f0_combo_, paths::gguf_files(models + QStringLiteral("/f0")), QStringLiteral("rmvpe.gguf"));
    fill(voice_combo_, paths::voice_models(), QString());

    tool_sep_ = paths::tool(QStringLiteral("rvc-sep"));
    tool_rvc_ = paths::tool(QStringLiteral("RVC-GGML"));
    tool_sox_ = QStandardPaths::findExecutable(QStringLiteral("sox"));
    tool_ffmpeg_ = QStandardPaths::findExecutable(QStringLiteral("ffmpeg"));

    log_view_->clear();
    if (root.isEmpty()) {
        log(QStringLiteral("Could not locate the models/ directory. Set the RVC_ROOT environment variable to the repository root."));
    } else {
        log(QStringLiteral("Repository root: %1").arg(root));
    }
    log(QStringLiteral("Tools: rvc-sep=%1 | RVC-GGML=%2 | sox=%3")
            .arg(tool_sep_.isEmpty() ? QStringLiteral("<missing>") : tool_sep_)
            .arg(tool_rvc_.isEmpty() ? QStringLiteral("<missing>") : tool_rvc_)
            .arg(tool_sox_.isEmpty() ? QStringLiteral("<missing>") : tool_sox_));
}

void SongCoverPage::refresh_models() {
    populate_models();
}

void SongCoverPage::browse_song() {
    const QString file = QFileDialog::getOpenFileName(
        this, QStringLiteral("Open song"), song_edit_->text(), QStringLiteral("Audio (*.wav *.flac *.mp3 *.ogg)"));
    if (file.isEmpty()) {
        return;
    }
    song_edit_->setText(file);
    if (output_edit_->text().isEmpty()) {
        const QFileInfo fi(file);
        output_edit_->setText(fi.absolutePath() + QStringLiteral("/") + fi.completeBaseName() +
                              QStringLiteral("_cover.wav"));
    }
}

void SongCoverPage::browse_output() {
    const QString file = QFileDialog::getSaveFileName(
        this, QStringLiteral("Output file"), output_edit_->text(), QStringLiteral("Audio (*.wav *.flac *.mp3 *.ogg)"));
    if (!file.isEmpty()) {
        output_edit_->setText(file);
    }
}

void SongCoverPage::log(const QString& text) {
    QString normalized = text;
    normalized.replace(QLatin1Char('\r'), QLatin1Char('\n'));
    const QStringList lines = normalized.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (const QString& line : lines) {
        log_view_->appendPlainText(line);
    }
    log_view_->verticalScrollBar()->setValue(log_view_->verticalScrollBar()->maximum());
}

void SongCoverPage::set_step_state(int index, const QChar& mark) {
    if (index < 0 || index >= steps_list_->count()) {
        return;
    }
    QListWidgetItem* item = steps_list_->item(index);
    QString text = item->text();
    text.remove(0, 2);  // strip the two-character prefix
    item->setText(QStringLiteral("%1 %2").arg(mark, text));
}

int SongCoverPage::total_pitch() const {
    return octaves_spin_->value() * 12 + semitones_spin_->value();
}

QString SongCoverPage::index_path_for(const QString& voice_model) const {
    const QStringList siblings = paths::gguf_files(QFileInfo(voice_model).absolutePath());
    for (const QString& s : siblings) {
        if (s.endsWith(QStringLiteral(".index.gguf")) &&
            QFileInfo(s).completeBaseName() == QFileInfo(voice_model).completeBaseName() + QStringLiteral(".index")) {
            return s;
        }
    }
    return QString();
}

QStringList SongCoverPage::rvc_args(const QString& in, const QString& out, bool auto_pitch) const {
    QStringList a;
    a << QStringLiteral("--input") << in
      << QStringLiteral("--output") << out
      << QStringLiteral("--model") << voice_combo_->currentData().toString();
    const QString idx = index_path_for(voice_combo_->currentData().toString());
    if (!idx.isEmpty()) {
        a << QStringLiteral("--index") << idx
          << QStringLiteral("--index-rate") << QString::number(index_rate_spin_->value());
    }
    a << QStringLiteral("--pitch") << QString::number(total_pitch())
      << QStringLiteral("--sid") << QString::number(sid_spin_->value())
      << QStringLiteral("--protect") << QString::number(protect_spin_->value())
      << QStringLiteral("--rms-mix-rate") << QString::number(rms_mix_spin_->value())
      << QStringLiteral("--hubert") << extractor_combo_->currentData().toString()
      << QStringLiteral("--rmvpe") << f0_combo_->currentData().toString()
      << QStringLiteral("--device") << device_combo_->currentData().toString();
    if (auto_pitch && proposed_pitch_check_->isChecked()) {
        a << QStringLiteral("--proposed-pitch")
          << QString::number(pitch_threshold_spin_->value(), 'f', 1);
    }
    return a;
}

void SongCoverPage::start_run() {
    if (running_) {
        return;
    }
    const QString song = song_edit_->text();
    const QString output = output_edit_->text();
    if (song.isEmpty() || !QFile::exists(song)) {
        QMessageBox::warning(this, QStringLiteral("Song cover"), QStringLiteral("Pick an existing song file first."));
        return;
    }
    if (output.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Song cover"), QStringLiteral("Pick an output file first."));
        return;
    }
    if (tool_sep_.isEmpty() || tool_rvc_.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Song cover"),
            QStringLiteral("rvc-sep / RVC-GGML were not found next to rvc-ui.\n"
                           "Set the RVC_TOOLS_DIR environment variable to the directory that contains them."));
        return;
    }
    if (vocal_sep_combo_->currentData().toString().isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Song cover"),
            QStringLiteral("No vocal separator model was found in models/roformers/vocal_separator."));
        return;
    }
    if (split_backup_check_->isChecked() && backup_sep_combo_->currentData().toString().isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Song cover"),
            QStringLiteral("Background split is enabled but no model was found in "
                           "models/roformers/backup_separator."));
        return;
    }
    if (dereverb_check_->isChecked() && dereverb_combo_->currentData().toString().isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Song cover"),
            QStringLiteral("De-reverb is enabled but no model was found in models/roformers/de-reverb."));
        return;
    }
    if (!QDir().mkpath(QFileInfo(output).absolutePath())) {
        QMessageBox::warning(this, QStringLiteral("Song cover"), QStringLiteral("Cannot create the output directory."));
        return;
    }

    work_dir_ = QFileInfo(output).absolutePath() + QStringLiteral("/") +
                QFileInfo(song).completeBaseName() + QStringLiteral("_work");
    QDir().mkpath(work_dir_);

    vocals_wav_ = work_dir_ + QStringLiteral("/vocals.wav");
    main_vocals_wav_ = vocals_wav_;
    background_wav_ = work_dir_ + QStringLiteral("/background_stem_1.wav");
    instrumental_wav_ = work_dir_ + QStringLiteral("/instrumental.wav");
    dereverb_wav_ = work_dir_ + QStringLiteral("/vocals_dereverbed.wav");
    conv_main_wav_ = work_dir_ + QStringLiteral("/vocals_converted.wav");
    conv_backup_wav_ = work_dir_ + QStringLiteral("/background_converted.wav");
    shifted_instrumental_wav_ = work_dir_ + QStringLiteral("/instrumental_shifted.wav");

    plan_.clear();
    steps_list_->clear();
    plan_ << StepKind::Separate;
    steps_list_->addItem(QStringLiteral("\u00b7 Separate vocals and instrumental"));
    if (split_backup_check_->isChecked()) {
        plan_ << StepKind::SplitBackup;
        steps_list_->addItem(QStringLiteral("\u00b7 Split background singers"));
    }
    if (dereverb_check_->isChecked()) {
        plan_ << StepKind::DeReverb;
        steps_list_->addItem(QStringLiteral("\u00b7 De-reverb main vocals"));
    }
    plan_ << StepKind::ConvertMain;
    steps_list_->addItem(QStringLiteral("\u00b7 Convert vocals (RVC)"));
    if (split_backup_check_->isChecked()) {
        plan_ << StepKind::ConvertBackup;
        steps_list_->addItem(QStringLiteral("\u00b7 Convert background vocals (RVC)"));
    }
    if (total_pitch() != 0) {
        plan_ << StepKind::ShiftInstrumental;
        steps_list_->addItem(QStringLiteral("\u00b7 Pitch-shift instrumental (%1 semitones)").arg(total_pitch()));
    }
    plan_ << StepKind::Mix;
    steps_list_->addItem(QStringLiteral("\u00b7 Mix and write output"));

    running_ = true;
    cancel_requested_ = false;
    step_index_ = -1;
    instrumental_shift_applied_ = false;
    run_timer_.start();
    progress_bar_->setValue(0);
    set_status(QStringLiteral("Starting..."));
    stop_playback();
    run_button_->setEnabled(false);
    cancel_button_->setEnabled(true);

    log(QStringLiteral("=== Song cover: %1 ===").arg(QFileInfo(song).fileName()));
    if (split_backup_check_->isChecked()) {
        const bool flip = flip_backup_check_->isChecked();
        log(QStringLiteral("Background split is enabled: main vocals = stem %1, background = stem %2.")
                .arg(flip ? 1 : 0).arg(flip ? 0 : 1));
    }
    if (flip_stems_check_->isChecked()) {
        log(QStringLiteral("Vocal stems are flipped: stem 1 = vocals, stem 0 = instrumental."));
    }
    if (dereverb_check_->isChecked()) {
        log(QStringLiteral("De-reverb: %1 -> main vocals before conversion.")
                .arg(display_name(dereverb_combo_->currentData().toString())));
    }
    run_step(0);
}

void SongCoverPage::run_step(int index) {
    step_index_ = index;
    const StepKind kind = plan_.at(index);
    set_step_state(index, QChar(0x25b8));  // running marker
    const QString label = steps_list_->item(index)->text().mid(2);
    set_status(QStringLiteral("Step %1/%2: %3").arg(index + 1).arg(plan_.size()).arg(label));
    progress_bar_->setValue(index * 100 / plan_.size());

    switch (kind) {
        case StepKind::Separate:
            // Clear outputs from a previous run: a stale _stem_1.wav would make a
            // single-stem run look like a two-stem run.
            QFile::remove(vocals_wav_);
            QFile::remove(work_dir_ + QStringLiteral("/vocals_stem_0.wav"));
            QFile::remove(work_dir_ + QStringLiteral("/vocals_stem_1.wav"));
            QFile::remove(work_dir_ + QStringLiteral("/vocals_flipped.wav"));
            start_process(tool_sep_, {vocal_sep_combo_->currentData().toString(),
                                      song_edit_->text(), vocals_wav_});
            break;
        case StepKind::SplitBackup:
            QFile::remove(work_dir_ + QStringLiteral("/background.wav"));
            QFile::remove(work_dir_ + QStringLiteral("/background_stem_0.wav"));
            QFile::remove(work_dir_ + QStringLiteral("/background_stem_1.wav"));
            start_process(tool_sep_, {backup_sep_combo_->currentData().toString(),
                                      vocals_wav_, work_dir_ + QStringLiteral("/background.wav")});
            break;
        case StepKind::DeReverb: {
            // The single-stem de-reverb model returns the dry (dereverbed) signal.
            // Clear stale outputs first so a failed run cannot reuse a previous take.
            QFile::remove(dereverb_wav_);
            QFile::remove(dereverb_wav_.left(dereverb_wav_.lastIndexOf(QLatin1Char('.'))) +
                          QStringLiteral("_stem_0.wav"));
            start_process(tool_sep_, {dereverb_combo_->currentData().toString(),
                                      main_vocals_wav_, dereverb_wav_});
            break;
        }
        case StepKind::ConvertMain:
            start_process(tool_rvc_, rvc_args(
                dereverb_check_->isChecked() ? dereverb_wav_ : main_vocals_wav_, conv_main_wav_, true));
            break;
        case StepKind::ConvertBackup:
            start_process(tool_rvc_, rvc_args(background_wav_, conv_backup_wav_, false));
            break;
        case StepKind::ShiftInstrumental: {
            // Only the manual semitone shift moves the instrumental; "auto pitch"
            // affects the converted voice alone.
            const int cents = total_pitch() * 100;
            instrumental_shift_applied_ = true;
            if (!tool_sox_.isEmpty()) {
                start_process(tool_sox_, {instrumental_wav_, shifted_instrumental_wav_,
                                          QStringLiteral("pitch"), QString::number(cents)});
            } else if (!tool_ffmpeg_.isEmpty()) {
                const double ratio = std::pow(2.0, total_pitch() / 12.0);
                const QString filter = QStringLiteral("asetrate=%1*%2,aresample=%1,atempo=%3")
                                           .arg(44100)
                                           .arg(ratio)
                                           .arg(1.0 / ratio);
                start_process(tool_ffmpeg_, {QStringLiteral("-y"), QStringLiteral("-i"),
                                             instrumental_wav_, QStringLiteral("-af"), filter,
                                             shifted_instrumental_wav_});
            } else {
                fail(QStringLiteral("Pitch shift is requested but neither sox nor ffmpeg was found; "
                                    "set the pitch to 0 or install sox."));
            }
            break;
        }
        case StepKind::Mix:
            run_mix();
            break;
    }
}

void SongCoverPage::start_process(const QString& program, const QStringList& args) {
    process_ = new QProcess(this);
    process_->setProgram(program);
    process_->setArguments(args);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, &QProcess::readyReadStandardOutput, this, [this] {
        log(QString::fromLocal8Bit(process_->readAllStandardOutput()));
    });
    connect(process_, &QProcess::finished, this, &SongCoverPage::on_process_finished);
    log(QStringLiteral(">> %1 %2").arg(program, args.join(QLatin1Char(' '))));
    process_->start();
    if (!process_->waitForStarted(5000)) {
        log(QStringLiteral("Failed to start: %1").arg(program));
        fail(QStringLiteral("Could not start %1").arg(QFileInfo(program).fileName()));
    }
}

void SongCoverPage::on_process_finished(int exit_code, int exit_status) {
    QProcess* proc = process_;
    process_ = nullptr;
    if (proc != nullptr) {
        proc->deleteLater();
    }
    if (cancel_requested_) {
        set_step_state(step_index_, QChar(0x2717));
        finish_run(false);
        log(QStringLiteral("Cancelled."));
        return;
    }
    if (exit_status != QProcess::NormalExit || exit_code != 0) {
        set_step_state(step_index_, QChar(0x2717));
        fail(QStringLiteral("Step failed with exit code %1. See the log above.").arg(exit_code));
        return;
    }

    complete_step();
}

void SongCoverPage::complete_step() {
    const StepKind kind = plan_.at(step_index_);
    set_step_state(step_index_, QChar(0x2713));
    progress_bar_->setValue((step_index_ + 1) * 100 / plan_.size());

    if (kind == StepKind::Separate && !prepare_instrumental()) {
        return;
    }
    if (kind == StepKind::SplitBackup && !prepare_split()) {
        return;
    }
    if (kind == StepKind::DeReverb && !resolve_dereverb_output()) {
        return;
    }

    if (step_index_ + 1 >= plan_.size()) {
        finish_run(true);
        return;
    }
    run_step(step_index_ + 1);
}

bool SongCoverPage::prepare_instrumental() {
    const QString stem0 = work_dir_ + QStringLiteral("/vocals_stem_0.wav");
    const QString stem1 = work_dir_ + QStringLiteral("/vocals_stem_1.wav");
    const QString plain = vocals_wav_;
    const bool flip = flip_stems_check_->isChecked();

    try {
        QApplication::setOverrideCursor(Qt::WaitCursor);
        if (QFile::exists(stem0) && QFile::exists(stem1)) {
            // Two-stem model: stem 0 = vocals, stem 1 = instrumental by convention;
            // some models swap them, which the "Flip stems" checkbox corrects.
            vocals_wav_ = flip ? stem1 : stem0;
            instrumental_wav_ = flip ? stem0 : stem1;
            log(QStringLiteral("Separator returned two stems: vocals = stem %1, instrumental = stem %2%3.")
                    .arg(flip ? 1 : 0).arg(flip ? 0 : 1)
                    .arg(flip ? QStringLiteral(" (flipped)") : QString()));
        } else if (QFile::exists(plain)) {
            // Vocals-only model (e.g. FV4): instrumental = song - vocals.
            log(QStringLiteral("Computing instrumental (song - vocals)..."));
            audio::Buffer song = audio::resample(audio::to_stereo(audio::load_audio(song_edit_->text().toStdString())), 44100);
            audio::Buffer voc = audio::resample(audio::to_stereo(audio::load_audio(plain.toStdString())), 44100);
            audio::Buffer inst = audio::subtract(song, voc);
            if (flip) {
                // Flipped single-stem model: treat the residual as the vocals.
                const QString flipped = work_dir_ + QStringLiteral("/vocals_flipped.wav");
                audio::save_wav(flipped.toStdString(), inst);
                vocals_wav_ = flipped;
                instrumental_wav_ = plain;
                log(QStringLiteral("Separator returned one stem; flipped: residual = vocals."));
            } else {
                audio::save_wav(instrumental_wav_.toStdString(), inst);
            }
        } else {
            QApplication::restoreOverrideCursor();
            fail(QStringLiteral("Separation produced no output file."));
            return false;
        }
        main_vocals_wav_ = vocals_wav_;
        QApplication::restoreOverrideCursor();
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        fail(QString::fromStdString(e.what()));
        return false;
    }
    return true;
}

bool SongCoverPage::prepare_split() {
    const QString stem0 = work_dir_ + QStringLiteral("/background_stem_0.wav");
    const QString stem1 = work_dir_ + QStringLiteral("/background_stem_1.wav");
    if (!QFile::exists(stem0) || !QFile::exists(stem1)) {
        fail(QStringLiteral("The background splitter returned a single stem; pick a 2-stem "
                            "(main vocals / background) model."));
        return false;
    }
    const bool flip = flip_backup_check_->isChecked();
    main_vocals_wav_ = flip ? stem1 : stem0;
    background_wav_ = flip ? stem0 : stem1;
    log(QStringLiteral("Background split: main vocals = stem %1, background = stem %2%3.")
            .arg(flip ? 1 : 0).arg(flip ? 0 : 1)
            .arg(flip ? QStringLiteral(" (flipped)") : QString()));
    return true;
}

bool SongCoverPage::resolve_dereverb_output() {
    const QString stem0 = dereverb_wav_.left(dereverb_wav_.lastIndexOf(QLatin1Char('.'))) +
                          QStringLiteral("_stem_0.wav");
    if (!QFile::exists(dereverb_wav_)) {
        if (QFile::exists(stem0)) {
            dereverb_wav_ = stem0;
        } else {
            fail(QStringLiteral("De-reverb produced no output file."));
            return false;
        }
    }
    log(QStringLiteral("De-reverb: dry vocals = %1").arg(QFileInfo(dereverb_wav_).fileName()));
    return true;
}

void SongCoverPage::run_mix() {
    try {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        const QString instrumental_file =
            (instrumental_shift_applied_ && QFile::exists(shifted_instrumental_wav_))
            ? shifted_instrumental_wav_ : instrumental_wav_;
        audio::Buffer instrumental = audio::resample(audio::to_stereo(audio::load_audio(instrumental_file.toStdString())), 44100);
        audio::Buffer main = audio::resample(audio::to_stereo(audio::load_audio(conv_main_wav_.toStdString())), 44100);

        std::vector<std::pair<audio::Buffer, float>> tracks;
        tracks.emplace_back(std::move(instrumental), audio::db_to_linear(static_cast<float>(inst_gain_spin_->value())));
        tracks.emplace_back(std::move(main), audio::db_to_linear(static_cast<float>(main_gain_spin_->value())));
        if (split_backup_check_->isChecked() && QFile::exists(conv_backup_wav_)) {
            audio::Buffer backup = audio::resample(audio::to_stereo(audio::load_audio(conv_backup_wav_.toStdString())), 44100);
            tracks.emplace_back(std::move(backup), audio::db_to_linear(static_cast<float>(backup_gain_spin_->value())));
        }

        audio::Buffer mixed = audio::mix(tracks);
        const float norm_db = audio::apply_peak_normalization(mixed);
        if (norm_db != 0.0f) {
            log(QStringLiteral("Peak normalization: %1%2 dB (to -1 dBFS)")
                    .arg(norm_db > 0.0f ? QStringLiteral("+") : QString())
                    .arg(static_cast<double>(norm_db), 0, 'f', 1));
        }
        const int out_sr = output_sr_combo_->currentData().toInt();
        if (out_sr != 44100) {
            mixed = audio::resample(mixed, out_sr);
        }

        const QString target = output_edit_->text();
        if (target.endsWith(QStringLiteral(".wav"), Qt::CaseInsensitive)) {
            audio::save_wav(target.toStdString(), mixed);
        } else {
            const QString wav_path = work_dir_ + QStringLiteral("/mix.wav");
            audio::save_wav(wav_path.toStdString(), mixed);
            QString err;
            if (!encode_to(wav_path, target, &err)) {
                QApplication::restoreOverrideCursor();
                set_step_state(step_index_, QChar(0x2717));
                fail(err);
                return;
            }
        }

        QApplication::restoreOverrideCursor();
        set_step_state(step_index_, QChar(0x2713));
        log(QStringLiteral("Wrote %1 (%2 s @ %3 Hz)")
                .arg(target)
                .arg(static_cast<double>(mixed.frames()) / mixed.sample_rate, 0, 'f', 2)
                .arg(mixed.sample_rate));
        load_output_for_playback(target);
        finish_run(true);
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        set_step_state(step_index_, QChar(0x2717));
        fail(QString::fromStdString(e.what()));
    }
}

bool SongCoverPage::encode_to(const QString& wav_path, const QString& target_path, QString* err) {
    if (tool_ffmpeg_.isEmpty()) {
        *err = QStringLiteral("ffmpeg was not found; it is required for FLAC/MP3/OGG output");
        return false;
    }
    QStringList args;
    args << QStringLiteral("-y") << QStringLiteral("-i") << wav_path;
    const QString lower = target_path.toLower();
    if (lower.endsWith(QStringLiteral(".flac"))) {
        args << QStringLiteral("-c:a") << QStringLiteral("flac");
    } else if (lower.endsWith(QStringLiteral(".mp3"))) {
        args << QStringLiteral("-c:a") << QStringLiteral("libmp3lame")
             << QStringLiteral("-q:a") << QStringLiteral("2");
    } else if (lower.endsWith(QStringLiteral(".ogg"))) {
        args << QStringLiteral("-c:a") << QStringLiteral("libvorbis")
             << QStringLiteral("-q:a") << QStringLiteral("5");
    } else {
        *err = QStringLiteral("Unsupported output extension (use .wav, .flac, .mp3 or .ogg)");
        return false;
    }
    args << target_path;
    log(QStringLiteral(">> %1 %2").arg(tool_ffmpeg_, args.join(QLatin1Char(' '))));

    QProcess proc;
    proc.setProcessChannelMode(QProcess::MergedChannels);
    proc.start(tool_ffmpeg_, args);
    if (!proc.waitForFinished(600000)) {
        *err = QStringLiteral("ffmpeg did not finish in time");
        return false;
    }
    const QByteArray out = proc.readAllStandardOutput();
    if (!out.isEmpty()) {
        log(QString::fromLocal8Bit(out));
    }
    if (proc.exitStatus() != QProcess::NormalExit || proc.exitCode() != 0) {
        *err = QStringLiteral("ffmpeg failed (exit code %1)").arg(proc.exitCode());
        return false;
    }
    return true;
}

void SongCoverPage::fail(const QString& message) {
    log(QStringLiteral("[error] %1").arg(message));
    set_status(QStringLiteral("Failed: %1").arg(message));
    finish_run(false);
}

void SongCoverPage::finish_run(bool success) {
    running_ = false;
    cancel_requested_ = false;
    run_button_->setEnabled(true);
    cancel_button_->setEnabled(false);
    if (success) {
        log(QStringLiteral("Song cover generated successfully."));
        log(QStringLiteral("Total time: %1 s (start -> file written)")
                .arg(run_timer_.elapsed() / 1000.0, 0, 'f', 1));
        progress_bar_->setValue(100);
        set_status(QStringLiteral("Done: %1").arg(QFileInfo(output_edit_->text()).fileName()));
    }
}

void SongCoverPage::cancel_run() {
    if (!running_) {
        return;
    }
    cancel_requested_ = true;
    if (process_ != nullptr) {
        process_->kill();
    }
}

void SongCoverPage::set_status(const QString& text) {
    status_label_->setText(text);
}

void SongCoverPage::show_log() {
    if (log_dialog_ == nullptr) {
        log_dialog_ = new QDialog(this);
        log_dialog_->setWindowTitle(QStringLiteral("Log"));
        log_dialog_->resize(780, 460);
        auto* layout = new QVBoxLayout(log_dialog_);
        layout->setContentsMargins(6, 6, 6, 6);
        layout->addWidget(log_view_);  // reparents the retained log view
    }
    log_dialog_->show();
    log_dialog_->raise();
    log_dialog_->activateWindow();
}

void SongCoverPage::toggle_playback() {
    if (player_ != nullptr) {
        player_->toggle_play();
    }
}

void SongCoverPage::stop_playback() {
    if (player_ != nullptr) {
        player_->stop();
    }
}

void SongCoverPage::on_slider_moved(int value) {
    const double duration = player_ != nullptr ? player_->duration_seconds() : 0.0;
    const double position = duration * value / position_slider_->maximum();
    time_label_->setText(QStringLiteral("%1 / %2").arg(
        format_time_seconds(position), format_time_seconds(duration)));
}

void SongCoverPage::on_slider_released() {
    slider_dragging_ = false;
    if (player_ != nullptr && player_->loaded() && player_->duration_seconds() > 0.0) {
        player_->seek_seconds(player_->duration_seconds() * position_slider_->value() /
                              position_slider_->maximum());
    }
}

void SongCoverPage::on_player_position(double position, double duration) {
    if (!slider_dragging_) {
        const double fraction = duration > 0.0 ? position / duration : 0.0;
        const QSignalBlocker blocker(position_slider_);
        position_slider_->setValue(static_cast<int>(fraction * position_slider_->maximum()));
    }
    time_label_->setText(QStringLiteral("%1 / %2").arg(
        format_time_seconds(position), format_time_seconds(duration)));
}

void SongCoverPage::on_player_state() {
    const bool playing = player_ != nullptr && player_->playing();
    play_button_->setText(playing ? QStringLiteral("Pause") : QStringLiteral("Play"));
    stop_button_->setEnabled(player_ != nullptr && player_->loaded());
}

void SongCoverPage::load_output_for_playback(const QString& path) {
    const bool ok = player_ != nullptr && player_->load(path);
    play_button_->setEnabled(ok);
    stop_button_->setEnabled(ok);
    position_slider_->setEnabled(ok);
    if (ok) {
        on_player_position(0.0, player_->duration_seconds());
        play_button_->setText(QStringLiteral("Play"));
    } else if (player_ != nullptr && player_->available()) {
        log(QStringLiteral("Playback: could not open %1").arg(path));
    }
}
