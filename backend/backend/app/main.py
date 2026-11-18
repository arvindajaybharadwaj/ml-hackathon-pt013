from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .candidates import NearestArcCandidateProvider, PreparedCandidateProvider
from .config import Settings
from .data import ReferenceRouteRepository, TripRepository
from .hmm import HMMConfig, HMMViterbiDecoder
from .predictor import DistanceFallbackPredictor, JoblibCandidatePredictor
from .schemas import EvaluationResponse, HealthResponse, PredictionResponse, TripSummary
from .services import EvaluationService, PredictionService


def build_services(settings: Settings):
    trips = TripRepository(settings.data_root, settings.partition)
    if settings.candidate_dir:
        candidate_provider = PreparedCandidateProvider(settings.candidate_dir)
    else:
        candidate_provider = NearestArcCandidateProvider(
            radius_m=settings.candidate_radius_m, max_candidates=settings.max_candidates
        )
    if settings.model_mode == "distance_fallback":
        if not settings.allow_dev_fallback:
            raise RuntimeError("Distance fallback is development-only; configure Person 2's model")
        predictor = DistanceFallbackPredictor(settings.emission_sigma_m)
    else:
        if not settings.model_path:
            raise RuntimeError("PATHSNAP_MODEL_PATH must point to Person 2's model artifact")
        predictor = JoblibCandidatePredictor(
            settings.model_path, settings.model_features,
            mode=settings.model_mode, positive_class=settings.positive_class,
        )
    decoder = HMMViterbiDecoder(HMMConfig(
        transition_sigma_base_m=settings.transition_sigma_base_m,
        transition_sigma_fraction=settings.transition_sigma_fraction,
        missing_state_penalty=settings.missing_state_penalty,
        unreachable_penalty=settings.unreachable_penalty,
    ))
    prediction = PredictionService(trips, candidate_provider, predictor, decoder)
    references = ReferenceRouteRepository(trips)
    evaluation = EvaluationService(prediction, references)
    return trips, candidate_provider, predictor, prediction, evaluation


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        trips, candidate_provider, predictor, prediction, evaluation = build_services(settings)
        app.state.trips = trips
        app.state.candidate_provider = candidate_provider
        app.state.predictor = predictor
        app.state.prediction = prediction
        app.state.evaluation = evaluation
        app.state.settings = settings
        yield

    application = FastAPI(
        title="PathSnap Prediction API",
        description="HMM/Viterbi route reconstruction backend for the Person 3 component.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    def state(request):
        if not hasattr(request.app.state, "prediction"):
            raise HTTPException(status_code=503, detail="Backend services are not initialised")
        return request.app.state

    @application.get("/", tags=["system"])
    def root():
        return {"name": "PathSnap Prediction API", "docs": "/docs", "health": "/health"}

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        s = state(request)
        return HealthResponse(
            status="ok",
            partition=settings.partition,
            model_loaded=s.predictor.mode != "development_fallback",
            predictor=s.predictor.name,
            predictor_mode=s.predictor.mode,
            candidate_provider=type(s.candidate_provider).__name__,
            data_root_exists=settings.data_root.exists(),
        )

    @application.get("/api/trips", tags=["trips"])
    def list_trips(request: Request):
        s = state(request)
        return {"partition": settings.partition, "trip_ids": s.trips.list_trip_ids()}

    @application.get("/api/trips/{trip_id}", response_model=TripSummary, tags=["trips"])
    def trip_metadata(trip_id: str, request: Request):
        s = state(request)
        try:
            return s.trips.metadata(trip_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/predict/{trip_id}", response_model=PredictionResponse, tags=["prediction"])
    def predict_trip(trip_id: str, request: Request):
        s = state(request)
        try:
            return s.prediction.predict(trip_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    @application.get("/api/evaluate/{trip_id}", response_model=EvaluationResponse, tags=["evaluation"])
    def evaluate_trip(trip_id: str, request: Request):
        s = state(request)
        try:
            return s.evaluation.evaluate(trip_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return application


app = create_app()
