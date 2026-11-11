// multi_step_page.cpp — multi-step generation implementation.

#include "multi_step_page.h"

#include "audio_utils.h"
#include "paths.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStandardPaths>
#include <QVBoxLayout>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

const QString kNone = QStringLiteral("__none__");
const QString kBrowse = QStringLiteral("__browse__");

// rvc-sep appends "_stem_<i>" before the extension for multi-stem models.
QString stem_path(const QString& out, int index) {
    const int dot = out.lastIndexOf(QLatin1Char('.'));
    return dot >= 0 ? out.left(dot) + QStringLiteral("_stem_%1").arg(index) + out.mid(dot)
                    : out + QStringLiteral("_stem_%1").arg(index);
}

// Sibling "<voice>.index.gguf" for a character model, if present.
QString index_for(const QString& voice_model) {
    const QStringList siblings = paths::gguf_files(QFileInfo(voice_model).absolutePath());
    for (const QString& s : siblings) {
        if (s.endsWith(QStringLiteral(".index.gguf")) &&
            QFileInfo(s).completeBaseName() ==
                QFileInfo(voice_model).completeBaseName() + QStringLiteral(".index")) {
            return s;
        }
    }
    return {};
}

} // namespace

// ---------------------------------------------------------------- registry

void SourceRegistry::add(const QString& name, const QString& path) {
    for (auto& entry : sources_) {
        if (entry.first == name) {
            entry.second = path;
            emit changed();
            return;
        }
    }
    sources_.append({name, path});
    emit changed();
}

// ---------------------------------------------------------------- step card

StepCard::StepCard(Kind kind, MultiStepPage* owner) : QGroupBox(owner), kind_(kind), owner_(owner) {
    build_ui();
    connect(&owner_->sources(), &SourceRegistry::changed, this, &StepCard::refresh_inputs);
    refresh_inputs();
}

void StepCard::build_ui() {
    auto* layout = new QVBoxLayout(this);
    auto* form = new QFormLayout();
    const QString models = paths::models_root();

    auto fill_models = [](QComboBox* combo, const QStringList& files, const QString& preferred) {
        for (const QString& f : files) {
            combo->addItem(QFileInfo(f).fileName(), f);
        }
        const QString pick = paths::preferred(files, preferred);
        if (!pick.isEmpty()) {
            const int idx = combo->findData(pick);
            if (idx >= 0) {
                combo->setCurrentIndex(idx);
            }
        }
    };
    auto add_input = [this, form](const QString& label) {
        auto* combo = new QComboBox(this);
        inputs_.append(combo);
        connect(combo, &QComboBox::activated, this, &StepCard::on_input_activated);
        form->addRow(label, combo);
        return combo;
    };
    auto add_output = [this, form](const QString& name, const QString& default_path) {
        auto* row = new QWidget(this);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        auto* edit = new QLineEdit(default_path, row);
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this, edit] {
            const QString file = QFileDialog::getSaveFileName(this, QStringLiteral("Output path"),
                edit->text(), QStringLiteral("Audio (*.wav)"));
            if (!file.isEmpty()) {
                edit->setText(file);
            }
        });
        const int index = output_edits_.size();
        auto* save = new QPushButton(QStringLiteral("Save as..."), row);
        connect(save, &QPushButton::clicked, this, [this, index] { save_output(index); });
        h->addWidget(edit, 1);
        h->addWidget(browse);
        h->addWidget(save);
        output_edits_.append(edit);
        output_names_.append(name);
        form->addRow(name, row);
    };

    switch (kind_) {
        case Kind::Roformer: {
            setTitle(QStringLiteral("Roformer separation / de-reverb (rvc-sep)"));
            add_input(QStringLiteral("Input"));
            model_combo_ = new QComboBox(this);
            auto add_models = [this](const QString& tag, const QString& dir) {
                for (const QString& f : paths::gguf_files(dir)) {
                    model_combo_->addItem(tag + QStringLiteral(" | ") + QFileInfo(f).fileName(), f);
                }
            };
            add_models(QStringLiteral("vocal"), models + QStringLiteral("/roformers/vocal_separator"));
            add_models(QStringLiteral("background"), models + QStringLiteral("/roformers/backup_separator"));
            add_models(QStringLiteral("de-reverb"), models + QStringLiteral("/roformers/de-reverb"));
            for (int i = 0; i < model_combo_->count(); ++i) {
                if (model_combo_->itemData(i).toString().contains(QStringLiteral("FV4"))) {
                    model_combo_->setCurrentIndex(i);
                    break;
                }
            }
            form->addRow(QStringLiteral("Model"), model_combo_);
            flip_check_ = new QCheckBox(QStringLiteral("Flip primary stem (stem 0 <-> stem 1)"), this);
            flip_check_->setToolTip(QStringLiteral(
                "Some separators swap the stem order. For single-stem models the primary output "
                "and its complement (input - output) are swapped instead."));
            form->addRow(QString(), flip_check_);

            roformer_default_paths_ = roformer_defaults(model_combo_->currentData().toString());
            add_output(QStringLiteral("Stem 0"), roformer_default_paths_.first);
            add_output(QStringLiteral("Stem 1"), roformer_default_paths_.second);
            output_edits_[0]->setToolTip(QStringLiteral(
                "Base output path. Multi-stem models write <base>_stem_0.wav / <base>_stem_1.wav."));
            output_edits_[1]->setToolTip(QStringLiteral(
                "Second stem for single-stem models: input - output (instrumental, or the "
                "removed reverb for de-reverb models)."));
            for (QLineEdit* edit : output_edits_) {
                connect(edit, &QLineEdit::textEdited, this, [this] { outputs_edited_ = true; });
            }
            connect(model_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                    this, &StepCard::on_model_changed);
            break;
        }
        case Kind::Convert: {
            setTitle(QStringLiteral("Convert voice (RVC-GGML)"));
            add_input(QStringLiteral("Input"));
            voice_combo_ = new QComboBox(this);
            fill_models(voice_combo_, paths::voice_models(), QString());
            form->addRow(QStringLiteral("Voice model"), voice_combo_);
            extractor_combo_ = new QComboBox(this);
            fill_models(extractor_combo_, paths::gguf_files(models + QStringLiteral("/extractors")),
                        QStringLiteral("contentvec"));
            form->addRow(QStringLiteral("Feature extractor"), extractor_combo_);
            f0_combo_ = new QComboBox(this);
            fill_models(f0_combo_, paths::gguf_files(models + QStringLiteral("/f0")), QStringLiteral("rmvpe.gguf"));
            form->addRow(QStringLiteral("F0 extractor"), f0_combo_);
            {
                auto* row = new QWidget(this);
                auto* h = new QHBoxLayout(row);
                h->setContentsMargins(0, 0, 0, 0);
                pitch_spin_ = new QSpinBox(row);
                pitch_spin_->setRange(-24, 24);
                pitch_spin_->setSuffix(QStringLiteral(" semitones"));
                auto_pitch_check_ = new QCheckBox(QStringLiteral("Auto pitch"), row);
                threshold_spin_ = new QDoubleSpinBox(row);
                threshold_spin_->setRange(50.0, 500.0);
                threshold_spin_->setValue(255.0);
                threshold_spin_->setSuffix(QStringLiteral(" Hz"));
                threshold_spin_->setEnabled(false);
                connect(auto_pitch_check_, &QCheckBox::toggled, threshold_spin_, &QDoubleSpinBox::setEnabled);
                h->addWidget(pitch_spin_);
                h->addWidget(auto_pitch_check_);
                h->addWidget(threshold_spin_);
                h->addStretch(1);
                form->addRow(QStringLiteral("Pitch"), row);
            }
            {
                auto* row = new QWidget(this);
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
                rms_mix_spin_ = new QDoubleSpinBox(row);
                rms_mix_spin_->setRange(0.0, 1.0);
                rms_mix_spin_->setSingleStep(0.05);
                rms_mix_spin_->setValue(0.0);
                h->addWidget(new QLabel(QStringLiteral("index"), row));
                h->addWidget(index_rate_spin_);
                h->addWidget(new QLabel(QStringLiteral("protect"), row));
                h->addWidget(protect_spin_);
                h->addWidget(new QLabel(QStringLiteral("rms mix"), row));
                h->addWidget(rms_mix_spin_);
                h->addStretch(1);
                form->addRow(QStringLiteral("RVC"), row);
            }
            {
                auto* row = new QWidget(this);
                auto* h = new QHBoxLayout(row);
                h->setContentsMargins(0, 0, 0, 0);
                sid_spin_ = new QSpinBox(row);
                sid_spin_->setRange(0, 500);
                device_combo_ = new QComboBox(row);
                device_combo_->addItem(QStringLiteral("GPU (Vulkan)"), 0);
                device_combo_->addItem(QStringLiteral("CPU"), 99);
                h->addWidget(new QLabel(QStringLiteral("speaker"), row));
                h->addWidget(sid_spin_);
                h->addWidget(new QLabel(QStringLiteral("device"), row));
                h->addWidget(device_combo_);
                h->addStretch(1);
                form->addRow(QStringLiteral("More"), row);
            }
            add_output(QStringLiteral("Output"), owner_->base_dir() + QStringLiteral("/converted.wav"));
            break;
        }
        case Kind::PitchShift: {
            setTitle(QStringLiteral("Pitch-shift audio (sox / ffmpeg)"));
            add_input(QStringLiteral("Input"));
            pitch_spin_ = new QSpinBox(this);
            pitch_spin_->setRange(-24, 24);
            pitch_spin_->setSuffix(QStringLiteral(" semitones"));
            form->addRow(QStringLiteral("Shift"), pitch_spin_);
            add_output(QStringLiteral("Output"), owner_->base_dir() + QStringLiteral("/shifted.wav"));
            break;
        }
        case Kind::Mix: {
            setTitle(QStringLiteral("Mix tracks (weighted sum, peak-safe)"));
            for (int i = 0; i < 3; ++i) {
                auto* row = new QWidget(this);
                auto* h = new QHBoxLayout(row);
                h->setContentsMargins(0, 0, 0, 0);
                auto* combo = new QComboBox(row);
                inputs_.append(combo);
                connect(combo, &QComboBox::activated, this, &StepCard::on_input_activated);
                auto* gain = new QDoubleSpinBox(row);
                gain->setRange(-40.0, 20.0);
                gain->setSuffix(QStringLiteral(" dB"));
                gain->setValue(0.0);
                gain_spins_.append(gain);
                h->addWidget(combo, 1);
                h->addWidget(gain);
                form->addRow(QStringLiteral("Track %1").arg(i + 1), row);
            }
            add_output(QStringLiteral("Output"), owner_->base_dir() + QStringLiteral("/mix.wav"));
            break;
        }
    }

    layout->addLayout(form);

    auto* bottom = new QHBoxLayout();
    run_button_ = new QPushButton(QStringLiteral("Run step"), this);
    connect(run_button_, &QPushButton::clicked, this, &StepCard::run);
    status_label_ = new QLabel(QStringLiteral("Idle"), this);
    bottom->addWidget(run_button_);
    bottom->addWidget(status_label_, 1);
    layout->addLayout(bottom);
}

void StepCard::refresh_inputs() {
    for (QComboBox* combo : inputs_) {
        const QString current = combo->currentData().toString();
        const QSignalBlocker blocker(combo);
        combo->clear();
        combo->addItem(QStringLiteral("<none>"), kNone);
        combo->addItem(QStringLiteral("<browse file...>"), kBrowse);
        for (const auto& entry : owner_->sources().sources()) {
            combo->addItem(entry.first, entry.second);
        }
        const int idx = combo->findData(current);
        combo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
}

void StepCard::set_status(const QString& text) {
    status_label_->setText(text);
}

QString StepCard::model_label() const {
    if (model_combo_ == nullptr || model_combo_->currentIndex() < 0) {
        return QStringLiteral("roformer");
    }
    return model_combo_->currentText();
}

// Model-qualified default output paths, so running the card with different
// models keeps every take and every registry entry distinct.
QPair<QString, QString> StepCard::roformer_defaults(const QString& model) const {
    QString tag = QFileInfo(model).completeBaseName();
    tag.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]+")), QStringLiteral("_"));
    if (tag.isEmpty()) {
        tag = QStringLiteral("model");
    }
    const QString base = owner_->base_dir() + QStringLiteral("/roformer_") + tag;
    return {base + QStringLiteral(".wav"), base + QStringLiteral("_complement.wav")};
}

void StepCard::on_model_changed() {
    if (output_edits_.size() < 2) {
        return;
    }
    const QPair<QString, QString> next = roformer_defaults(model_combo_->currentData().toString());
    if (!outputs_edited_) {
        output_edits_[0]->setText(next.first);
        output_edits_[1]->setText(next.second);
    }
    roformer_default_paths_ = next;
    resolved_.clear();
}

QString StepCard::input_path(int index) const {
    if (index < 0 || index >= inputs_.size()) {
        return {};
    }
    const QVariant data = inputs_.at(index)->currentData();
    if (!data.isValid()) {
        return {};
    }
    const QString path = data.toString();
    return (path == kNone || path == kBrowse) ? QString() : path;
}

QString StepCard::output_path(int index) const {
    if (kind_ == Kind::Roformer && index >= 0 && index < resolved_.size() &&
        !resolved_.at(index).isEmpty() && QFile::exists(resolved_.at(index))) {
        return resolved_.at(index);
    }
    return index >= 0 && index < output_edits_.size() ? output_edits_.at(index)->text() : QString();
}

void StepCard::on_input_activated(int index) {
    auto* combo = qobject_cast<QComboBox*>(sender());
    if (combo == nullptr) {
        return;
    }
    if (combo->itemData(index).toString() != kBrowse) {
        return;
    }
    const QString file = QFileDialog::getOpenFileName(this, QStringLiteral("Open audio"), QString(),
        QStringLiteral("Audio (*.wav *.flac *.mp3 *.ogg)"));
    if (file.isEmpty()) {
        combo->setCurrentIndex(0);
        return;
    }
    owner_->sources().add(QStringLiteral("File: ") + QFileInfo(file).fileName(), file);
    const int idx = combo->findData(file);
    if (idx >= 0) {
        combo->setCurrentIndex(idx);
    }
}

void StepCard::run() {
    if (running_) {
        return;
    }

    if (kind_ == Kind::Mix) {
        set_status(QStringLiteral("Mixing..."));
        try {
            std::vector<std::pair<audio::Buffer, float>> tracks;
            for (int i = 0; i < inputs_.size(); ++i) {
                const QString path = input_path(i);
                if (path.isEmpty()) {
                    continue;
                }
                audio::Buffer buffer =
                    audio::resample(audio::to_stereo(audio::load_audio(path.toStdString())), 44100);
                tracks.emplace_back(std::move(buffer),
                                    audio::db_to_linear(static_cast<float>(gain_spins_.at(i)->value())));
            }
            if (tracks.empty()) {
                throw std::runtime_error("select at least one input track");
            }
            audio::Buffer mixed = audio::mix(tracks);
            audio::apply_peak_normalization(mixed);
            audio::save_wav(output_path(0).toStdString(), mixed);
        } catch (const std::exception& e) {
            set_status(QStringLiteral("Failed: %1").arg(QString::fromStdString(e.what())));
            return;
        }
        register_outputs();
        set_status(QStringLiteral("Done"));
        return;
    }

    QString program;
    QStringList args;
    QString error;
    if (!build_invocation(&program, &args, &error)) {
        set_status(QStringLiteral("Failed: %1").arg(error));
        return;
    }

    if (kind_ == Kind::Roformer) {
        // A previous run (possibly with another model) may have left
        // <base>_stem_N.wav / the complement behind; stale files would be
        // mistaken for this run's outputs.
        if (output_edits_.size() >= 2) {
            const QString base = output_edits_.at(0)->text();
            QFile::remove(base);
            QFile::remove(stem_path(base, 0));
            QFile::remove(stem_path(base, 1));
            QFile::remove(output_edits_.at(1)->text());
        }
        resolved_.clear();
    }

    running_ = true;
    run_button_->setEnabled(false);
    set_status(QStringLiteral("Running..."));

    process_ = new QProcess(this);
    process_->setProgram(program);
    process_->setArguments(args);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, &QProcess::readyReadStandardOutput, this, [this] {
        QString text = QString::fromLocal8Bit(process_->readAllStandardOutput());
        text.replace(QLatin1Char('\r'), QLatin1Char('\n'));
        const QStringList lines = text.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
        if (!lines.isEmpty()) {
            const QString last = lines.last().trimmed();
            if (!last.isEmpty()) {
                set_status(last.right(110));
            }
        }
    });
    connect(process_, &QProcess::finished, this, &StepCard::on_run_finished);
    process_->start();
    if (!process_->waitForStarted(5000)) {
        set_status(QStringLiteral("Failed to start %1").arg(QFileInfo(program).fileName()));
        running_ = false;
        run_button_->setEnabled(true);
        process_->deleteLater();
        process_ = nullptr;
    }
}

void StepCard::on_run_finished(int exit_code, int exit_status) {
    QProcess* proc = process_;
    process_ = nullptr;
    if (proc != nullptr) {
        proc->deleteLater();
    }
    running_ = false;
    run_button_->setEnabled(true);

    if (exit_status != QProcess::NormalExit || exit_code != 0) {
        set_status(QStringLiteral("Failed (exit %1)").arg(exit_code));
        return;
    }
    try {
        handle_success();
    } catch (const std::exception& e) {
        set_status(QStringLiteral("Failed: %1").arg(QString::fromStdString(e.what())));
        return;
    }
    register_outputs();
    set_status(QStringLiteral("Done"));
}

bool StepCard::build_invocation(QString* program, QStringList* args, QString* error) {
    const QString tool_sep = paths::tool(QStringLiteral("rvc-sep"));
    const QString tool_rvc = paths::tool(QStringLiteral("RVC-GGML"));

    switch (kind_) {
        case Kind::Roformer: {
            if (tool_sep.isEmpty()) {
                *error = QStringLiteral("rvc-sep not found (set RVC_TOOLS_DIR)");
                return false;
            }
            const QString input = input_path(0);
            if (input.isEmpty()) {
                *error = QStringLiteral("pick an input");
                return false;
            }
            *program = tool_sep;
            *args = {model_combo_->currentData().toString(), input, output_path(0)};
            return true;
        }
        case Kind::Convert: {
            if (tool_rvc.isEmpty()) {
                *error = QStringLiteral("RVC-GGML not found (set RVC_TOOLS_DIR)");
                return false;
            }
            const QString input = input_path(0);
            if (input.isEmpty()) {
                *error = QStringLiteral("pick an input");
                return false;
            }
            const QString voice = voice_combo_->currentData().toString();
            *program = tool_rvc;
            *args << QStringLiteral("--input") << input
                  << QStringLiteral("--output") << output_path(0)
                  << QStringLiteral("--model") << voice;
            const QString idx = index_for(voice);
            if (!idx.isEmpty()) {
                *args << QStringLiteral("--index") << idx
                      << QStringLiteral("--index-rate") << QString::number(index_rate_spin_->value());
            }
            *args << QStringLiteral("--pitch") << QString::number(pitch_spin_->value())
                  << QStringLiteral("--sid") << QString::number(sid_spin_->value())
                  << QStringLiteral("--protect") << QString::number(protect_spin_->value())
                  << QStringLiteral("--rms-mix-rate") << QString::number(rms_mix_spin_->value())
                  << QStringLiteral("--hubert") << extractor_combo_->currentData().toString()
                  << QStringLiteral("--rmvpe") << f0_combo_->currentData().toString()
                  << QStringLiteral("--device") << device_combo_->currentData().toString();
            if (auto_pitch_check_->isChecked()) {
                *args << QStringLiteral("--proposed-pitch")
                      << QString::number(threshold_spin_->value(), 'f', 1);
            }
            return true;
        }
        case Kind::PitchShift: {
            const int semitones = pitch_spin_->value();
            if (semitones == 0) {
                *error = QStringLiteral("set a non-zero shift");
                return false;
            }
            const QString input = input_path(0);
            if (input.isEmpty()) {
                *error = QStringLiteral("pick an input");
                return false;
            }
            const QString sox = QStandardPaths::findExecutable(QStringLiteral("sox"));
            if (!sox.isEmpty()) {
                *program = sox;
                *args = {input, output_path(0), QStringLiteral("pitch"), QString::number(semitones * 100)};
                return true;
            }
            const QString ffmpeg = QStandardPaths::findExecutable(QStringLiteral("ffmpeg"));
            if (!ffmpeg.isEmpty()) {
                const double ratio = std::pow(2.0, semitones / 12.0);
                *program = ffmpeg;
                *args << QStringLiteral("-y") << QStringLiteral("-i") << input
                      << QStringLiteral("-af")
                      << QStringLiteral("asetrate=44100*%1,aresample=44100,atempo=%2")
                             .arg(ratio)
                             .arg(1.0 / ratio)
                      << output_path(0);
                return true;
            }
            *error = QStringLiteral("neither sox nor ffmpeg was found");
            return false;
        }
        case Kind::Mix:
            break;  // handled in-process by run()
    }
    return false;
}

void StepCard::handle_success() {
    switch (kind_) {
        case Kind::Roformer: {
            const QString out = output_path(0);
            const QString stem0 = stem_path(out, 0);
            const QString stem1 = stem_path(out, 1);
            QString primary;
            QString secondary;
            if (QFile::exists(stem0) && QFile::exists(stem1)) {
                // Two-stem model: use the model's own pair.
                primary = stem0;
                secondary = stem1;
            } else {
                // Single-stem model: the model's output is the primary stem; the
                // complement (input - output) is the other. For a vocal separator
                // that is vocals/instrumental, for a de-reverb model dry/reverb.
                const QString single = QFile::exists(out) ? out : (QFile::exists(stem0) ? stem0 : QString());
                if (single.isEmpty()) {
                    throw std::runtime_error("rvc-sep produced no output");
                }
                primary = single;
                secondary = output_path(1);
                if (secondary.isEmpty() || secondary == primary) {
                    throw std::runtime_error("pick a different path for the second stem");
                }
                const QString input = input_path(0);
                if (input.isEmpty()) {
                    throw std::runtime_error("input audio went missing");
                }
                audio::Buffer in = audio::resample(
                    audio::to_stereo(audio::load_audio(input.toStdString())), 44100);
                audio::Buffer p = audio::resample(
                    audio::to_stereo(audio::load_audio(primary.toStdString())), 44100);
                audio::save_wav(secondary.toStdString(), audio::subtract(in, p));
            }
            if (flip_check_ != nullptr && flip_check_->isChecked()) {
                std::swap(primary, secondary);
            }
            resolved_ = {primary, secondary};
            return;
        }
        default:
            if (!QFile::exists(output_path(0))) {
                throw std::runtime_error("step produced no output");
            }
            return;
    }
}

void StepCard::register_outputs() {
    switch (kind_) {
        case Kind::Roformer: {
            const QString label = QStringLiteral("Roformer (%1)").arg(model_label());
            owner_->sources().add(label + QStringLiteral(": stem 0"), output_path(0));
            owner_->sources().add(label + QStringLiteral(": stem 1"), output_path(1));
            break;
        }
        case Kind::Convert:
            owner_->sources().add(QStringLiteral("Convert: output"), output_path(0));
            break;
        case Kind::PitchShift:
            owner_->sources().add(QStringLiteral("Pitch-shift: output"), output_path(0));
            break;
        case Kind::Mix:
            owner_->sources().add(QStringLiteral("Mix: output"), output_path(0));
            break;
    }
}

void StepCard::save_output(int index) {
    const QString source = output_path(index);
    if (source.isEmpty() || !QFile::exists(source)) {
        set_status(QStringLiteral("No output to save yet"));
        return;
    }
    const QString target = QFileDialog::getSaveFileName(this, QStringLiteral("Save output"),
        QFileInfo(source).fileName(), QStringLiteral("Audio (*.wav *.flac *.mp3 *.ogg)"));
    if (target.isEmpty()) {
        return;
    }
    if (QFile::exists(target)) {
        QFile::remove(target);
    }
    set_status(QFile::copy(source, target) ? QStringLiteral("Saved: %1").arg(target)
                                           : QStringLiteral("Copy failed: %1").arg(target));
}

// ---------------------------------------------------------------- page

MultiStepPage::MultiStepPage(QWidget* parent) : QWidget(parent) {
    const QString root = paths::root();
    base_dir_ = root.isEmpty() ? QDir::tempPath() + QStringLiteral("/rvc_multistep")
                               : root + QStringLiteral("/work/multistep");
    QDir().mkpath(base_dir_);

    auto* layout = new QVBoxLayout(this);
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    auto* content = new QWidget(scroll);
    auto* content_layout = new QVBoxLayout(content);

    content_layout->addWidget(new StepCard(StepCard::Kind::Roformer, this));
    content_layout->addWidget(new StepCard(StepCard::Kind::Convert, this));
    content_layout->addWidget(new StepCard(StepCard::Kind::PitchShift, this));
    content_layout->addWidget(new StepCard(StepCard::Kind::Mix, this));
    content_layout->addStretch(1);
    scroll->setWidget(content);
    layout->addWidget(scroll);
}
