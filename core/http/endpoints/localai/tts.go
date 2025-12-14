package localai

import (
	"encoding/binary"
	"fmt"
	"path/filepath"

	"github.com/labstack/echo/v4"
	"github.com/mudler/LocalAI/core/backend"
	"github.com/mudler/LocalAI/core/config"
	"github.com/mudler/LocalAI/core/http/middleware"
	"github.com/mudler/LocalAI/pkg/grpc/proto"
	"github.com/mudler/LocalAI/pkg/model"

	"github.com/mudler/LocalAI/core/schema"
	"github.com/rs/zerolog/log"

	"github.com/mudler/LocalAI/pkg/utils"
)

// TTSEndpoint is the OpenAI Speech API endpoint https://platform.openai.com/docs/api-reference/audio/createSpeech
//
//		@Summary	Generates audio from the input text.
//	 	@Accept json
//	 	@Produce audio/x-wav
//		@Param		request	body		schema.TTSRequest	true	"query params"
//		@Success	200		{string}	binary				"generated audio/wav file"
//		@Router		/v1/audio/speech [post]
//		@Router		/tts [post]
func TTSEndpoint(cl *config.ModelConfigLoader, ml *model.ModelLoader, appConfig *config.ApplicationConfig) echo.HandlerFunc {
	return func(c echo.Context) error {
		input, ok := c.Get(middleware.CONTEXT_LOCALS_KEY_LOCALAI_REQUEST).(*schema.TTSRequest)
		if !ok || input.Model == "" {
			return echo.ErrBadRequest
		}

		cfg, ok := c.Get(middleware.CONTEXT_LOCALS_KEY_MODEL_CONFIG).(*config.ModelConfig)
		if !ok || cfg == nil {
			return echo.ErrBadRequest
		}

		log.Debug().Str("model", input.Model).Msg("LocalAI TTS Request received")

		if cfg.Backend == "" && input.Backend != "" {
			cfg.Backend = input.Backend
		}

		if input.Language != "" {
			cfg.Language = input.Language
		}

		if input.Voice != "" {
			cfg.Voice = input.Voice
		}

		filePath, _, err := backend.ModelTTS(input.Input, cfg.Voice, cfg.Language, ml, appConfig, *cfg)
		if err != nil {
			return err
		}

		// Convert generated file to target format
		filePath, err = utils.AudioConvert(filePath, input.Format)
		if err != nil {
			return err
		}

		return c.Attachment(filePath, filepath.Base(filePath))
	}
}

// TTSStreamEndpoint streams TTS audio as chunked HTTP response
//
//	@Summary	Generates streaming audio from the input text.
//	@Accept		json
//	@Produce	audio/wav
//	@Param		request	body		schema.TTSRequest	true	"query params"
//	@Success	200		{string}	binary				"streaming audio/wav data"
//	@Router		/tts/stream [post]
//	@Router		/v1/audio/speech/stream [post]
func TTSStreamEndpoint(cl *config.ModelConfigLoader, ml *model.ModelLoader, appConfig *config.ApplicationConfig) echo.HandlerFunc {
	return func(c echo.Context) error {
		input, ok := c.Get(middleware.CONTEXT_LOCALS_KEY_LOCALAI_REQUEST).(*schema.TTSRequest)
		if !ok || input.Model == "" {
			return echo.ErrBadRequest
		}

		cfg, ok := c.Get(middleware.CONTEXT_LOCALS_KEY_MODEL_CONFIG).(*config.ModelConfig)
		if !ok || cfg == nil {
			return echo.ErrBadRequest
		}

		log.Debug().Str("model", input.Model).Msg("LocalAI TTS Stream Request received")

		if cfg.Backend == "" && input.Backend != "" {
			cfg.Backend = input.Backend
		}

		if input.Language != "" {
			cfg.Language = input.Language
		}

		if input.Voice != "" {
			cfg.Voice = input.Voice
		}

		// Set streaming headers
		c.Response().Header().Set("Content-Type", "audio/wav")
		c.Response().Header().Set("Transfer-Encoding", "chunked")
		c.Response().Header().Set("Cache-Control", "no-cache")
		c.Response().Header().Set("Connection", "keep-alive")

		ctx := c.Request().Context()
		writer := c.Response().Writer

		// Track if we've written the WAV header
		headerWritten := false

		err := backend.ModelTTSStream(
			ctx,
			input.Input,
			cfg.Voice,
			cfg.Language,
			ml,
			appConfig,
			*cfg,
			func(chunk *proto.TTSStreamChunk) error {
				if chunk.Error != nil && *chunk.Error != "" {
					return fmt.Errorf("TTS error: %s", *chunk.Error)
				}

				if !headerWritten && len(chunk.Audio) > 0 {
					// Write WAV header for streaming (use max size placeholder)
					wavHeader := createWAVHeader(0xFFFFFFFF, chunk.SampleRate, 1, 16)
					if _, err := writer.Write(wavHeader); err != nil {
						return err
					}
					headerWritten = true
				}

				if len(chunk.Audio) > 0 {
					if _, err := writer.Write(chunk.Audio); err != nil {
						return err
					}
					c.Response().Flush()
				}

				return nil
			},
		)

		if err != nil {
			if !headerWritten {
				return err
			}
			log.Error().Err(err).Msg("TTS streaming error")
		}

		return nil
	}
}

// createWAVHeader creates a WAV header for streaming audio
func createWAVHeader(dataSize uint32, sampleRate int32, numChannels int16, bitsPerSample int16) []byte {
	header := make([]byte, 44)
	byteRate := sampleRate * int32(numChannels) * int32(bitsPerSample) / 8
	blockAlign := numChannels * bitsPerSample / 8

	// RIFF header
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], dataSize+36)
	copy(header[8:12], "WAVE")

	// fmt chunk
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], uint16(numChannels))
	binary.LittleEndian.PutUint32(header[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(header[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(header[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(header[34:36], uint16(bitsPerSample))

	// data chunk
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], dataSize)

	return header
}
